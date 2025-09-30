import requests
import os

BASE_URL = "http://127.0.0.1:8000"

def test_reset():
    print("--- Testing /reset ---")
    try:
        response = requests.post(f"{BASE_URL}/reset")
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("----------------------\n")

def test_ingest():
    print("--- Testing /ingest ---")
    try:
        # Create a dummy text file
        with open("test.txt", "w") as f:
            f.write("This is a test file for the ingest endpoint. It contains some text about a user.")

        # Use an existing image file from the assets folder
        image_path = os.path.join("assets", "user.png")

        with open('test.txt', 'rb') as txt_f:
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}. Skipping image ingestion.")
                files_to_upload = [('files', ('test.txt', txt_f, 'text/plain'))]
                response = requests.post(f"{BASE_URL}/ingest", files=files_to_upload)
            else:
                with open(image_path, 'rb') as img_f:
                    files_to_upload = [
                        ('files', ('test.txt', txt_f, 'text/plain')),
                        ('files', (os.path.basename(image_path), img_f, 'image/png'))
                    ]
                    response = requests.post(f"{BASE_URL}/ingest", files=files_to_upload)
            
            response.raise_for_status()
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists("test.txt"):
            os.remove("test.txt")
    print("-----------------------\n")

def test_chat():
    print("--- Testing /chat ---")
    try:
        print("Testing chat with a query about the ingested file...")
        payload = {"query": "What is the test file about?"}
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    print("---------------------\n")

def test_transcribe():
    print("--- Testing /transcribe ---")
    print("Skipping /transcribe test: No dummy audio file available.")
    print("To test this endpoint, place an audio file (e.g., 'test_audio.mp3') in the same directory as this script and uncomment the code below.")
    # try:
    #     audio_file_path = 'test_audio.mp3'
    #     if os.path.exists(audio_file_path):
    #         with open(audio_file_path, "rb") as f:
    #             files = {'file': (os.path.basename(audio_file_path), f, 'audio/mpeg')}
    #             response = requests.post(f"{BASE_URL}/transcribe", files=files)
    #             response.raise_for_status()
    #             print(f"Status Code: {response.status_code}")
    #             print(f"Response: {response.json()}")
    #     else:
    #         print(f"Audio file not found at {audio_file_path}")
    # except requests.exceptions.RequestException as e:
    #     print(f"Error: {e}")
    print("-------------------------\n")

def test_chat_audio():
    print("--- Testing /chat-audio ---")
    print("Skipping /chat-audio test: No dummy audio file available for query.")
    print("To test this endpoint, place an audio file (e.g., 'query.mp3') in this directory and use a tool like Postman or curl to send it to the /chat-audio endpoint.")
    print("Example with curl:")
    print("curl -X POST -F 'file=@query.mp3' http://127.0.0.1:8000/chat-audio")
    print("---------------------------\n")


if __name__ == "__main__":
    print("Starting backend tests...")
    # Ensure the backend server is running before executing this script.

    # 1. Reset the state to start clean
    test_reset()

    # 2. Ingest new data for the chat test
    test_ingest()

    # 3. Chat about the ingested data
    test_chat()

    # 4. Test transcription (placeholder)
    test_transcribe()

    # 5. Test audio chat (placeholder)
    test_chat_audio()

    print("All tests finished.")