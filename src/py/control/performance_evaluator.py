import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict


class PerformanceEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
        self.frame_count = 0
        self.detection_times = []
        self.tracking_stats = {
            'total_frames': 0,
            'successful_tracks': 0,
            'target_losses': 0,
            'tracking_accuracy': []
        }

        base_output_dir = os.path.dirname(__file__)
        self.output_dir = os.path.join(f'/../../{base_output_dir}', 'performance_reports')
        os.makedirs(self.output_dir, exist_ok=True)

    def start_evaluation(self):
        self.start_time = time.time()

    def log_detection(self, detection_time, target_info, ground_truth=None):
        self.frame_count += 1
        self.detection_times.append(detection_time)

        if ground_truth and target_info:
            iou = self._calculate_iou(target_info, ground_truth)
            self.metrics['detection_accuracy'].append(iou)

        if self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                fps = self.frame_count / elapsed_time
                self.metrics['fps'].append(
                    fps if len(self.metrics['fps']) < 10 else np.mean(self.metrics['fps'][-9:] + [fps]))

    def log_tracking(self, target_info):
        self.tracking_stats['total_frames'] += 1

        if target_info:
            offset_distance = np.hypot(target_info['x_offset'], target_info['y_offset'])
            self.tracking_stats['tracking_accuracy'].append(offset_distance)

            if offset_distance > 50:
                self.tracking_stats['target_losses'] += 1
            else:
                self.tracking_stats['successful_tracks'] += 1
        else:
            self.tracking_stats['target_losses'] += 1

    def generate_visualization_report(self):
        plt.style.use('seaborn')
        self._plot_metric(self.detection_times, 'Detection Time Distribution', 'Time (s)', 'detection_time.png')
        self._plot_metric(self.metrics['fps'], 'FPS Over Time', 'Frame', 'fps_trend.png', line=True)
        self._plot_metric(self.tracking_stats['tracking_accuracy'], 'Tracking Accuracy', 'Offset Distance',
                          'tracking_accuracy.png')

    def _plot_metric(self, data, title, xlabel, filename, line=False):
        if not data:
            return
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True) if not line else plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def save_report(self):
        self.generate_visualization_report()
        report_path = os.path.join(self.output_dir, 'performance_report.txt')
        with open(report_path, 'w') as f:
            f.write("Performance Report\n" + "=" * 30 + "\n\n")
            f.write(
                f"Avg Detection Time: {np.mean(self.detection_times):.4f}s\n" if self.detection_times else "No valid detection data\n")
            f.write(f"Avg FPS: {np.mean(self.metrics['fps']):.2f}\n" if self.metrics['fps'] else "No valid FPS data\n")
            f.write(f"Total Frames: {self.frame_count}\n")
            success_rate = (self.tracking_stats['successful_tracks'] / max(1,
                                                                           self.tracking_stats['total_frames'])) * 100
            f.write(f"Tracking Success Rate: {success_rate:.2f}%\n")
            f.write(f"Avg Tracking Accuracy: {np.mean(self.tracking_stats['tracking_accuracy']):.4f}\n" if
                    self.tracking_stats['tracking_accuracy'] else "No valid tracking accuracy data\n")
        print(f"Report saved in: {self.output_dir}")

    @staticmethod
    def _calculate_iou(detection, ground_truth):
        d_x1, d_y1, d_x2, d_y2 = detection.get('bbox', [0, 0, 0, 0])
        g_x1, g_y1, g_x2, g_y2 = ground_truth
        x_left, y_top = max(d_x1, g_x1), max(d_y1, g_y1)
        x_right, y_bottom = min(d_x2, g_x2), min(d_y2, g_y2)
        intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        union = ((d_x2 - d_x1) * (d_y2 - d_y1)) + ((g_x2 - g_x1) * (g_y2 - g_y1)) - intersection
        return intersection / union if union else 0.0
