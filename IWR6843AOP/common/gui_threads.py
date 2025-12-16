# General Library Imports
import numpy as np
import time
import os
from collections import deque
from typing import Optional, Dict, Any

# PyQt imports
from PySide2.QtCore import QThread, Signal, QMutexLocker, QMutex, QWaitCondition

import pyqtgraph as pg

# Local Imports
from gui_parser import UARTParser
from gui_common import *
from graph_utilities import *

import joblib
from tcn import TCN
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import load_model
from sklearn.cluster import DBSCAN

# Logger
import logging
log = logging.getLogger(__name__)

# Model Configuration
TIMESTEPS = 30
VERSION = "v4"
MODEL_FILE = f"{VERSION}_Model.h5"
SCALER_FILE = f"{VERSION}_Scaler.pkl"

# Classifier Configurables
MAX_NUM_TRACKS = 20

# Expected ranges for coloring
SNR_EXPECTED_MIN = 5
SNR_EXPECTED_MAX = 40
SNR_EXPECTED_RANGE = SNR_EXPECTED_MAX - SNR_EXPECTED_MIN
DOPPLER_EXPECTED_MIN = -30
DOPPLER_EXPECTED_MAX = 30
DOPPLER_EXPECTED_RANGE = DOPPLER_EXPECTED_MAX - DOPPLER_EXPECTED_MIN

# Color modes
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'

# Magic Numbers for Target Index TLV
TRACK_INDEX_WEAK_SNR = 253
TRACK_INDEX_BOUNDS = 254
TRACK_INDEX_NOISE = 255

#region Helper Classes
class ExponentialSmoothing(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        x = inputs
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        F = tf.shape(x)[2]

        ta = tf.TensorArray(dtype=x.dtype, size=T)
        prev = x[:, 0, :]
        ta = ta.write(0, prev)

        def body(t, prev, ta):
            curr = x[:, t, :]
            smoothed = self.alpha * curr + (1 - self.alpha) * prev
            ta = ta.write(t, smoothed)
            return t + 1, smoothed, ta

        def cond(t, prev, ta):
            return t < T

        _, _, ta = tf.while_loop(cond, body, [1, prev, ta])
        out = ta.stack()
        out = tf.transpose(out, [1, 0, 2])
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

class PositionalEncoding(layers.Layer):
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[-1]

        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
        )

        angle_rads = position * div_term
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        pos_encoding = pos_encoding[:, :d_model]
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        return inputs + pos_encoding

def merge_padding_and_attention_mask(inputs, padding_mask, attention_mask):
    input_mask = None
    if hasattr(inputs, "_keras_mask") and inputs._keras_mask is not None:
        input_mask = tf.cast(inputs._keras_mask, tf.bool)

    final_mask = None
    if padding_mask is not None:
        padding_mask = tf.cast(padding_mask, tf.bool)
        final_mask = padding_mask[:, tf.newaxis, :]

    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, tf.bool)
        final_mask = attention_mask

    if final_mask is None and input_mask is not None:
        final_mask = input_mask[:, tf.newaxis, :]

    return final_mask

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0.0,
        activation="relu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = tf.keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = hidden_dim // self.num_heads

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self.att_norm = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.att_dropout = tf.keras.layers.Dropout(self.dropout)

        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ffn_dense1 = tf.keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.ffn_dense2 = tf.keras.layers.Dense(
            hidden_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout)

    def call(
        self,
        inputs,
        padding_mask=None,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
    ):
        mask = merge_padding_and_attention_mask(inputs, padding_mask, attention_mask)

        x = inputs
        residual = x

        if self.normalize_first:
            x = self.att_norm(x)

        att_output, att_scores = self.mha(
            query=x,
            value=x,
            attention_mask=mask,
            return_attention_scores=True,
            training=training,
        )

        x = self.att_dropout(att_output, training=training)
        x = x + residual

        if not self.normalize_first:
            x = self.att_norm(x)

        residual = x

        if self.normalize_first:
            x = self.ffn_norm(x)

        x = self.ffn_dense1(x)
        x = self.ffn_dense2(x)
        x = self.ffn_dropout(x, training=training)
        x = x + residual

        if not self.normalize_first:
            x = self.ffn_norm(x)

        return (x, att_scores) if return_attention_scores else x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "intermediate_dim": self.intermediate_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "activation": tf.keras.activations.serialize(self.activation),
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "normalize_first": self.normalize_first,
        })
        return cfg
#endregion




#region Parse Data
class parseUartThread(QThread):
    fin = Signal(dict)
    batchReady = Signal(np.ndarray)

    def __init__(self, uParser):
        super().__init__()
        self.parser = uParser
        self.running = True
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.outputDir = f'./dataset/{self.timestamp}'
        os.makedirs(self.outputDir, exist_ok=True)
        
        # Performance optimization: batch CSV writes
        self.csv_buffer = []
        self.csv_buffer_size = 10
        self.last_csv_write = time.time()
        self.csv_write_interval = 1.0

    def run(self):
        while self.running:
            try:
                # Parse data
                if self.parser.parserType == "SingleCOMPort":
                    outputDict = self.parser.readAndParseUartSingleCOMPort()
                else:
                    outputDict = self.parser.readAndParseUartDoubleCOMPort()

                if not outputDict:
                    time.sleep(0.001)
                    continue

                self.fin.emit(outputDict)

                # Buffered CSV writing
                if self.parser.saveBinary == 1:
                    frameJSON = {
                        'frameData': outputDict,
                        'timestamp': time.time() * 1000
                    }
                    self.csv_buffer.append(frameJSON)
                    
                    now = time.time()
                    if len(self.csv_buffer) >= self.csv_buffer_size or (now - self.last_csv_write) >= self.csv_write_interval:
                        self._flush_csv_buffer()
                        self.last_csv_write = now

                # Process for ML pipeline
                frame_points = self.processFrame(outputDict)
                if frame_points is not None:
                    self.batchReady.emit(frame_points)

            except Exception as e:
                logging.error(f"[parseUartThread] Error: {e}")
                continue

    def _flush_csv_buffer(self):
        """Write buffered CSV data to disk"""
        if not self.csv_buffer:
            return
        
        try:
            csvFilePath = f'{self.outputDir}/dataset.csv'
            for frameJSON in self.csv_buffer:
                self.parser.saveDataToCsv(csvFilePath, frameJSON)
            self.csv_buffer.clear()
        except Exception as e:
            logging.error(f"[parseUartThread] CSV write error: {e}")

    def processFrame(self, frameData: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract point cloud for ML pipeline"""
        try:
            pointCloud = frameData.get("pointCloud")
            
            if pointCloud is None or len(pointCloud) == 0:
                return None
            
            if not isinstance(pointCloud, np.ndarray):
                pointCloud = np.asarray(pointCloud, dtype=np.float32)
            
            if pointCloud.size == 0 or pointCloud.ndim != 2 or pointCloud.shape[1] < 5:
                return None
            
            frame_points = pointCloud[:, :5]
            
            if frame_points.shape[0] < 10:
                return None

            return frame_points

        except Exception as e:
            logging.error(f"[processFrame] Error: {e}")
            return None

    def stop(self):
        self._flush_csv_buffer()
        self.running = False
        self.wait(2000)
#endregion




#region Preprocessing
class preprocessThread(QThread):
    preprocessedReady = Signal(np.ndarray)

    def __init__(self, scaler_path=f"IWR/common/Scalers/{SCALER_FILE}", timesteps=TIMESTEPS):
        super().__init__()
        
        try:
            self.scaler = joblib.load(scaler_path)
            logging.info("[preprocessThread] Scaler loaded successfully")
        except Exception as e:
            logging.error(f"[preprocessThread] Failed to load scaler: {e}")
            self.scaler = None
        
        self.timesteps = timesteps
        self.running = True

        self.queue = deque(maxlen=20)
        self.frames_buffer = deque(maxlen=timesteps * 2)

        self.eps_value = 0.3
        self.min_points = 10
        self.dbscan = DBSCAN(eps=self.eps_value, min_samples=self.min_points)

        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        
        self.frame_skip_count = 0

    def addBatch(self, points_np: np.ndarray):
        """Thread-safe method to add new frame data"""
        with QMutexLocker(self.mutex):
            if len(self.queue) >= self.queue.maxlen - 1:
                self.frame_skip_count += 1
                if self.frame_skip_count % 100 == 0:
                    logging.warning(f"[preprocessThread] Queue full, skipped {self.frame_skip_count} frames")
            self.queue.append(points_np)
            self.wait_condition.wakeOne()

    def run(self):
        if self.scaler is None:
            logging.error("[preprocessThread] No scaler loaded. Exiting.")
            return

        while self.running:
            self.mutex.lock()
            while not self.queue and self.running:
                self.wait_condition.wait(self.mutex, 50)

            if not self.running:
                self.mutex.unlock()
                break

            frame_points = self.queue.popleft() if self.queue else None
            self.mutex.unlock()

            if frame_points is None or len(frame_points) == 0:
                continue

            try:
                centroid = self._processCluster(frame_points)
                if centroid is None:
                    continue

                self.frames_buffer.append(centroid)

                if len(self.frames_buffer) < self.timesteps:
                    continue

                window = np.array(list(self.frames_buffer)[-self.timesteps:], dtype=np.float32)
                scaled_window = self.scaler.transform(window)

                self.preprocessedReady.emit(scaled_window)

            except Exception as e:
                logging.error(f"[preprocessThread] Error: {e}")
                continue

    def _processCluster(self, frame_points: np.ndarray) -> Optional[np.ndarray]:
        """Extract largest cluster and return centroid"""
        try:
            xyz = frame_points[:, :3]

            if len(xyz) < self.min_points:
                return None

            labels = self.dbscan.fit_predict(xyz)

            valid_labels = labels[labels != -1]
            if len(valid_labels) == 0:
                return None

            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            largest_cluster_label = unique_labels[np.argmax(counts)]
            
            cluster_mask = (labels == largest_cluster_label)
            cluster_points = frame_points[cluster_mask]

            if len(cluster_points) == 0:
                return None

            return np.mean(cluster_points, axis=0, dtype=np.float32)

        except Exception as e:
            logging.error(f"[_processCluster] Error: {e}")
            return None

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
            self.wait_condition.wakeAll()
        self.wait(2000)
#endregion




#region Prediction
class predictThread(QThread):
    predictionReady = Signal(str)

    def __init__(
        self,
        model_path=f"IWR/common/Models/{MODEL_FILE}",
        vote_window=5,
        fall_threshold=0.95,
        fall_confirm_frames=7,      # <-- NEW: required consecutive fall frames
        emit_rate_hz=10,
        parent=None
    ):
        super().__init__(parent)

        self.model = None
        try:
            self.model = load_model(
                model_path,
                custom_objects={
                    'PositionalEncoding': PositionalEncoding,
                    'TCN': TCN,
                    'TransformerEncoder': TransformerEncoder
                },
                compile=False
            )
            logging.info("[predictThread] Model loaded successfully")
        except Exception as e:
            logging.error(f"[predictThread] Failed to load model: {e}")

        # Warm-up pass to initialize TensorFlow graph
        try:
            dummy = np.zeros((1, 30, 5), dtype=np.float32)  # match model input
            _ = self.model.predict(dummy, verbose=0)
            logging.info("[predictThread] Warm-up forward pass complete")
        except Exception as e:
            logging.error(f"[predictThread] Warm-up failed: {e}")

        # Queue for preprocessed data
        self.queue = deque(maxlen=15)
        self.running = True

        # Voting / smoothing
        self.vote_window = vote_window
        self.fall_buffer = deque(maxlen=vote_window)
        self.fall_threshold = fall_threshold

        # NEW: fall confirmation system
        self.fall_confirm_frames = fall_confirm_frames
        self.consecutive_fall_frames = 0
        self.last_state = "Others"

        self.emit_interval = 1.0 / emit_rate_hz
        self.last_emit_time = 0

        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()

        self.prediction_count = 0
        self.avg_inference_time = 0

    def addPreprocessed(self, batch: np.ndarray):
        """Thread-safe method to add preprocessed data"""
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 2:
            batch = np.expand_dims(batch, axis=0)

        with QMutexLocker(self.mutex):
            # Avoid queue blocking by discarding oldest when full
            if len(self.queue) == self.queue.maxlen:
                self.queue.popleft()
            self.queue.append(batch)
            self.wait_condition.wakeOne()

    def run(self):
        if self.model is None:
            logging.error("[predictThread] No model loaded. Exiting.")
            return

        while self.running:
            self.mutex.lock()
            while not self.queue and self.running:
                self.wait_condition.wait(self.mutex, 50)

            if not self.running:
                self.mutex.unlock()
                break

            batch = self.queue.popleft() if self.queue else None
            self.mutex.unlock()

            if batch is None or batch.size == 0:
                continue

            try:
                # Measure inference latency
                start_time = time.time()
                predictions = self.model.predict(batch, verbose=0)[0]
                inference_time = time.time() - start_time

                # Rolling average inference time
                self.prediction_count += 1
                self.avg_inference_time = (
                    self.avg_inference_time * (self.prediction_count - 1) +
                    inference_time
                ) / self.prediction_count

                prob_others = float(predictions[0])
                prob_fall   = float(predictions[1])

                # Rolling smoothing for fall probability
                self.fall_buffer.append(prob_fall)
                avg_fall = float(np.mean(self.fall_buffer))

                # Frame-level fall detection
                is_fall_frame = avg_fall >= self.fall_threshold

                # Consecutive fall frame counter
                if is_fall_frame:
                    self.consecutive_fall_frames += 1
                else:
                    self.consecutive_fall_frames = 0

                # Confirm fall only after required fall frames
                if self.consecutive_fall_frames >= self.fall_confirm_frames:
                    state = "FALL!"
                    confidence = avg_fall
                else:
                    state = "Others"
                    confidence = prob_others

                # Emit prediction based on rate limit
                now = time.time()
                if now - self.last_emit_time >= self.emit_interval:
                    result = f"{state} {confidence:.2f}"
                    self.predictionReady.emit(result)
                    self.last_emit_time = now

            except Exception as e:
                logging.error(f"[predictThread] Prediction error: {e}")
                continue

    def stop(self):
        logging.info(
            f"[predictThread] Avg inference time: {self.avg_inference_time*1000:.2f}ms"
        )
        with QMutexLocker(self.mutex):
            self.running = False
            self.wait_condition.wakeAll()
        self.wait(2000)
#endregion




#region Etc.
class sendCommandThread(QThread):
    done = Signal()

    def __init__(self, uParser, command):
        QThread.__init__(self)
        self.parser = uParser
        self.command = command

    def run(self):
        self.parser.sendLine(self.command)
        self.done.emit()


class updateQTTargetThread3D(QThread):
    done = Signal()

    def __init__(self, pointCloud, targets, scatter, pcplot, numTargets, ellipsoids, coords, colorGradient=None, classifierOut=[], zRange=[-3, 3], pointColorMode="", drawTracks=True, trackColorMap=None, pointBounds={'enabled': False}):
        QThread.__init__(self)
        self.pointCloud = pointCloud
        self.targets = targets
        self.scatter = scatter
        self.pcplot = pcplot
        self.colorArray = ('r', 'g', 'b', 'w')
        self.numTargets = numTargets
        self.ellipsoids = ellipsoids
        self.coordStr = coords
        self.classifierOut = classifierOut
        self.zRange = zRange
        self.colorGradient = colorGradient
        self.pointColorMode = pointColorMode
        self.drawTracks = drawTracks
        self.trackColorMap = trackColorMap
        self.pointBounds = pointBounds
        # This ignores divide by 0 errors when calculating the log2
        np.seterr(divide='ignore')

    def drawTrack(self, track, trackColor):
        # Get necessary track data
        tid = int(track[0])
        x = track[1]
        y = track[2]
        z = track[3]

        track = self.ellipsoids[tid]
        mesh = getBoxLinesCoords(x, y, z)
        track.setData(pos=mesh, color=trackColor, width=2,
                      antialias=True, mode='lines')
        track.setVisible(True)

    # Return transparent color if pointBounds is enabled and point is outside pointBounds
    # Otherwise, color the point depending on which color mode we are in
    def getPointColors(self, i):
        if (self.pointBounds['enabled']):
            xyz_coords = self.pointCloud[i, 0:3]
            if (xyz_coords[0] < self.pointBounds['minX']
                        or xyz_coords[0] > self.pointBounds['maxX']
                        or xyz_coords[1] < self.pointBounds['minY']
                        or xyz_coords[1] > self.pointBounds['maxY']
                        or xyz_coords[2] < self.pointBounds['minZ']
                        or xyz_coords[2] > self.pointBounds['maxZ']
                    ) :
                return pg.glColor((0, 0, 0, 0))

        # Color the points by their SNR
        if (self.pointColorMode == COLOR_MODE_SNR):
            snr = self.pointCloud[i, 4]
            # SNR value is out of expected bounds, make it white
            if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((snr-SNR_EXPECTED_MIN)/SNR_EXPECTED_RANGE))

        # Color the points by their Height
        elif (self.pointColorMode == COLOR_MODE_HEIGHT):
            zs = self.pointCloud[i, 2]

            # Points outside expected z range, make it white
            if (zs < self.zRange[0]) or (zs > self.zRange[1]):
                return pg.glColor('w')
            else:
                colorRange = self.zRange[1]+abs(self.zRange[0])
                zs = self.zRange[1] - zs
                return pg.glColor(self.colorGradient.getColor(abs(zs/colorRange)))

        # Color Points by their doppler
        elif (self.pointColorMode == COLOR_MODE_DOPPLER):
            doppler = self.pointCloud[i, 3]
            # Doppler value is out of expected bounds, make it white
            if (doppler < DOPPLER_EXPECTED_MIN) or (doppler > DOPPLER_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((doppler-DOPPLER_EXPECTED_MIN)/DOPPLER_EXPECTED_RANGE))

        # Color the points by their associate track
        elif (self.pointColorMode == COLOR_MODE_TRACK):
            trackIndex = int(self.pointCloud[i, 6])
            # trackIndex of 253, 254, or 255 indicates a point isn't associated to a track, so check for those magic numbers here
            if (trackIndex == TRACK_INDEX_WEAK_SNR or trackIndex == TRACK_INDEX_BOUNDS or trackIndex == TRACK_INDEX_NOISE):
                return pg.glColor('w')
            else:
                # Catch any errors that may occur if track or point index go out of bounds
                try:
                    return self.trackColorMap[trackIndex]
                except Exception as e:
                    log.error(e)
                    return pg.glColor('w')

        # Unknown Color Option, make all points green
        else:
            return pg.glColor('g')

    def run(self):

        # if self.pointCloud is None or len(self.pointCloud) == 0:
        #     print("Point Cloud is empty or None.")
        # else:
        #     print("Point Cloud Shape:", self.pointCloud.shape)

        # Clear all previous targets
        for e in self.ellipsoids:
            if (e.visible()):
                e.hide()
        try:
            # Create a list of just X, Y, Z values to be plotted
            if (self.pointCloud is not None):
                toPlot = self.pointCloud[:, 0:3]
                # print("Data for Visualization:", toPlot)

                # Determine the size of each point based on its SNR
                with np.errstate(divide='ignore'):
                    size = np.log2(self.pointCloud[:, 4])

                # Each color is an array of 4 values, so we need an numPoints*4 size 2d array to hold these values
                pointColors = np.zeros((self.pointCloud.shape[0], 4))

                # Set the color of each point
                for i in range(self.pointCloud.shape[0]):
                    pointColors[i] = self.getPointColors(i)

                # Plot the points
                self.scatter.setData(pos=toPlot, color=pointColors, size=size)
                # Debugging
                # print("Pos Data for Visualization:", toPlot)
                # print("Color Data for Visualization:", pointColors)
                # print("Size Data for Visualization:", size)

                # Make the points visible
                self.scatter.setVisible(True)
            else:
                # Make the points invisible if none are detected.
                self.scatter.setVisible(False)
        except Exception as e:
            log.error(
                "Unable to draw point cloud, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")

        # Graph the targets
        try:
            if (self.drawTracks):
                if (self.targets is not None):
                    for track in self.targets:
                        trackID = int(track[0])
                        trackColor = self.trackColorMap[trackID]
                        self.drawTrack(track, trackColor)
        except:
            log.error(
                "Unable to draw all tracks, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")
        self.done.emit()

    def stop(self):
        self.terminate()
#endregion