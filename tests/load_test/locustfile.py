import time
from locust import HttpUser, task, between
import load_test
import time
import random

class QuickstartUser(HttpUser):
    wait_time = between(1, 5)
    test_df = load_test.build_dataframe(10)

    @task
    def hello_world(self):      
        for col in self.test_df.columns:
            for message in self.test_df[col]:               
                self.client.post("/message", json={"conv_text":message, "chat_id": int(random.choice(self.test_df.columns))}, headers={'Connection':'close'})
                time.sleep(1)


