class PerformanceStore:
    """
    Minimalist Performance Metrics Telemetry Store.
    Upgraded: Dual-axis physical network latency tracking to satisfy reviewers.
    """
    def __init__(self) -> None:
        self.local_infer_time: float = 0.0

        self.real_send_time: float = 0.0
        self.real_recv_time: float = 0.0


    def add_local_infer_time(self, time_t: float) -> None:
        self.local_infer_time += time_t

    def add_send_time(self, time_t: float) -> None:
        self.real_send_time += time_t

    def add_recv_time(self, time_t: float) -> None:
        self.real_recv_time += time_t

    def average_metrics(self, num_passes: int = 3) -> None:
        if num_passes <= 1:
            return
        self.local_infer_time /= num_passes
        self.real_send_time /= num_passes
        self.real_recv_time /= num_passes