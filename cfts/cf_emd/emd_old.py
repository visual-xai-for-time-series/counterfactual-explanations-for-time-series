from typing import List

import numpy as np
from scipy import signal
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cfts.cf_emd.emd_old as emd_old

class Chunk:
    def __init__(self, data, label, imfs=None):
        self.data = np.array(data)
        self.label = label
        self.f, self.Pxx_spec = signal.welch(self.data, scaling='spectrum')
        self.imfs = emd_old.sift.sift(self.data).T if imfs is None else np.array(imfs)
        self.compute_ip_if_ia()

    @classmethod
    def from_json(cls, json_input):
        return cls(data=json_input["data"], label=json_input["label"], imfs=json_input["imfs"])

    def compute_ip_if_ia(self):
        IP, IF, IA = emd_old.spectra.frequency_transform(self.imfs, sample_rate=1.0, method='hilbert')
        self.average_ip = np.mean(IP, axis=0)
        self.average_if = np.mean(IF, axis=0)
        self.average_ia = np.mean(IA, axis=0)

    def ip_if_ia_distance(self, other: "Chunk", imf_index: int):
        vec1 = np.array([self.average_ia[imf_index], self.average_if[imf_index], self.average_ip[imf_index]])
        vec2 = np.array([other.average_ia[imf_index], other.average_if[imf_index], other.average_ip[imf_index]])
        return np.linalg.norm(vec1 - vec2)

    def to_json(self):
        return {
            "data": self.data.tolist(),
            "imfs": self.imfs.tolist(),
            "label": self.label,
            "powerSpectralDensity": self.Pxx_spec.tolist(),
            "f": self.f.tolist()
        }


def chunk_from_imfs(imfs, target_class) -> Chunk:
    data = np.sum(imfs, axis=0)
    chunk = Chunk(data, target_class)
    chunk.imfs = imfs
    chunk.compute_ip_if_ia()
    return chunk


def compute_fingerprint_histogram(data):
    f, Pxx_spec = signal.welch(data, scaling='spectrum')
    return Pxx_spec

def compute_time_series_variance(time_series_ds) -> float:
    histograms = []
    for ts in time_series_ds:
        histograms.append(compute_fingerprint_histogram(ts))
    X = np.array(histograms)
    X = X / X.sum(axis=1, keepdims=True)
    mean_hist = X.mean(axis=0)
    var_between = float(np.mean((X - mean_hist) ** 2))
    return var_between


# def train_classifier(histograms, labels):
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(labels)
#     model = make_pipeline(
#         StandardScaler(),
#         KNeighborsClassifier(n_neighbors=1)
#     )
#     model.fit(histograms, y)
#
#     # Compute training accuracy
#     preds = model.predict(histograms)
#     acc = accuracy_score(y, preds)
#     print(f"Training accuracy: {acc:.2f}")
#
#     return model, label_encoder


class CounterfactualGenerator:
    def __init__(self, chunks: List[Chunk], source_idx: int, target_class: int, model, method):
        count_target_class = len([c for c in chunks if c.label == target_class])

        if source_idx < 0 or source_idx >= len(chunks):
            raise ValueError(f"source_idx must be within [1, {len(chunks) - 1}]")

        if count_target_class == 0:
            raise ValueError(f"No points in target class found")

        if method not in ["distance", "variance"]:
            raise ValueError("Method must be either distance or variance")

        self.chunks = chunks
        self.source: Chunk = chunks[source_idx]
        self.source_idx = source_idx
        self.target_class = target_class
        self.native_guides_count = 1
        self.native_guide_idx = self.get_native_guide_idx()
        self.native_guide = self.chunks[self.native_guide_idx]
        self.interpolation_weights = self.init_interpolation_weights()
        self.model = model
        self.method = method
        self.cf_path: List[Chunk] = [self.source]

    def get_native_guide_idx(self) -> int:
        distances = []
        source_hist = self.source.Pxx_spec
        for idx, c in enumerate(self.chunks):
            if c.label != self.target_class:
                continue
            hist = c.Pxx_spec
            dist = jensenshannon(source_hist, hist)
            distances.append((dist, idx))
        distances.sort(key=lambda s: s[0])
        return distances[0][1]

    def get_interpolate(self):
        interpolated_imfs = []
        for idx, w in enumerate(self.interpolation_weights):
            if idx < len(self.source.imfs) and idx < len(self.native_guide.imfs):
                interpolated = w["w_source"] * self.source.imfs[idx] + w["w_target"] * self.native_guide.imfs[idx] * \
                               w["w_target"]
                interpolated_imfs.append(interpolated)
        interpolated_imfs = np.array(interpolated_imfs)
        interpolate: Chunk = chunk_from_imfs(interpolated_imfs, "cf")
        interpolate.label = self.model.predict(interpolate.Pxx_spec.reshape(1, -1))
        return interpolate

    def get_imf_level_distances(self):
        interpolate_imfs = self.get_interpolate().imfs
        target_imfs = self.native_guide.imfs
        max_idx = len(self.interpolation_weights)
        distances = []
        for idx in range(max_idx):
            if len(interpolate_imfs) <= idx or len(target_imfs) <= idx:
                distances.append(0)
                continue
            imf1 = interpolate_imfs[idx]
            imf2 = target_imfs[idx]
            hist1 = compute_fingerprint_histogram(imf1)
            hist2 = compute_fingerprint_histogram(imf2)
            distances.append(jensenshannon(hist1, hist2))
        distances = np.array(distances)
        distances = distances / np.max(distances)
        return distances

    def get_imf_level_variance_distances(self, imf_level):
        chunks_source = [c.imfs[imf_level] for c in self.chunks if
                         c.label == self.source.label and len(c.imfs) > imf_level]
        chunks_target = [c.imfs[imf_level] for c in self.chunks if
                         c.label == self.target_class and len(c.imfs) > imf_level]
        var_source = compute_time_series_variance(chunks_source)
        var_target = compute_time_series_variance(chunks_target)
        return np.abs(var_source - var_target)

    def get_step_distances(self):
        if self.method == "distance":
            return self.get_imf_level_distances()
        else:
            var_distances = []
            for idx, w in enumerate(self.interpolation_weights):
                var_distances.append(self.get_imf_level_variance_distances(idx))
            return np.array(var_distances) / np.max(var_distances)

    def step_interpolation_weigths(self, step=0.05):
        # TODO implement advanced step mechanism based on variation
        distances = self.get_step_distances()
        for idx, w in enumerate(self.interpolation_weights):
            weight_step = step * distances[idx]
            w["w_target"] = min(1, w["w_target"] + weight_step)
            w["w_source"] = max(0, w["w_source"] - weight_step)

    def step_until_class_flip(self, step=0.05):
        i = 0
        while True:
            print(i)
            self.step_interpolation_weigths(step=step)
            interpolate = self.get_interpolate()
            self.cf_path.append(interpolate)
            if interpolate.label == self.native_guide.label:
                break
            i += 1

    def init_interpolation_weights(self):
        max_imf_index = max(len(self.native_guide.imfs), len(self.source.imfs))
        weights = []
        for _ in range(max_imf_index):
            weights.append({"w_source": 1, "w_target": 0})
        return weights

    def to_json(self):
        return {
            "source_id": self.source_idx,
            "source_class": self.source.label,
            "target_class": self.target_class,
            "native_guide_id": self.native_guide_idx
        }
