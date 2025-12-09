from locust import HttpUser, task, between
import random

class GestureRecognitionUser(HttpUser):
    """
    Simulates users sending EMG data to the gesture recognition server.
    Run with: locust -f locustfile.py --host=http://localhost:8000
    """
    # Wait between 0.1 and 0.5 seconds between requests (simulating real-time EMG data)
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Check if server is alive
        self.client.get("/ping")
    
    @task
    def predict_batch(self):
        """Test batch prediction endpoint"""
        batch_size = random.randint(5, 20)  # Simulate batches of 5-20 samples
        batch = []
        for _ in range(batch_size):
            batch.append({
                "EMG1": random.randint(-50, 50),
                "EMG2": random.randint(-50, 50),
                "EMG3": random.randint(-50, 50),
                "EMG4": random.randint(-50, 50),
                "EMG5": random.randint(-50, 50),
                "EMG6": random.randint(-50, 50),
                "EMG7": random.randint(-50, 50),
                "EMG8": random.randint(-50, 50),
            })
        
        payload = {"batch": batch}
        with self.client.post("/predict_batch", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == batch_size:
                    response.success()
                else:
                    response.failure("Invalid batch response")
            else:
                response.failure(f"Got status code {response.status_code}")


class ComparisonUser(HttpUser):
    """
    Dedicated load test to compare single vs batch performance.
    Run with: locust -f locustfile.py ComparisonUser --host=http://localhost:8000
    """
    wait_time = between(0.05, 0.1)
    
    @task
    def compare_endpoints(self):
        """Send 10 single requests vs 1 batch of 10"""
        # This task helps you see the overhead difference
        batch = []
        for _ in range(10):
            batch.append({
                "EMG1": random.randint(-50, 50),
                "EMG2": random.randint(-50, 50),
                "EMG3": random.randint(-50, 50),
                "EMG4": random.randint(-50, 50),
                "EMG5": random.randint(-50, 50),
                "EMG6": random.randint(-50, 50),
                "EMG7": random.randint(-50, 50),
                "EMG8": random.randint(-50, 50),
            })
        
        # Batch request
        self.client.post("/predict_batch", json={"batch": batch}, name="/predict_batch (10 samples)")
