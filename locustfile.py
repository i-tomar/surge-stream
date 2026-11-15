"""Load generation against the prediction API."""

from locust import HttpUser, between, task
import random


class PredictUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(8)
    def predict_zone(self) -> None:
        zone = random.randint(1, 263)
        horizon = 15 if random.random() < 0.7 else 30
        self.client.get(f"/api/predict/{zone}?horizon={horizon}", name="/api/predict/[zone]")

    @task(1)
    def zones(self) -> None:
        self.client.get("/api/zones")
