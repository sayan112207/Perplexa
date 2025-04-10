from locust import HttpUser, task, between

class ChatbotUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def send_query(self):
        payload = {"messages": [{"role": "user", "content": "What is AI?"}]}
        headers = {"Content-Type": "application/json"}

        response = self.client.post("/", json=payload, headers=headers)
        assert response.status_code == 200

# Run using: locust -f tests/locust_test.py --host=http://localhost:5500
# Things to Change:
# If the API runs on another server, update --host=http://your-server-url.

# Run Load Test
# streamlit run app.py
# locust -finally --host=http://localhost:5500