import arxiv
from run import analyze_papers_with_gemini
import run

# Override history load to return empty set so it forces analysis
run.load_history = lambda: set()

# We will test on a known paper from history
import time
time.sleep(5)
test_id = "2604.18122v1"

print(f"Fetching paper {test_id} for test...")
client = arxiv.Client()
search = arxiv.Search(id_list=[test_id])
papers = list(client.results(search))

if papers:
    print(f"Found paper: {papers[0].title}")
    # Change output dir for testing so we don't mess up real reports
    import os
    run.OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_reports')
    run.IMAGES_DIR = os.path.join(run.OUTPUT_DIR, 'images')
    if not os.path.exists(run.OUTPUT_DIR):
        os.makedirs(run.OUTPUT_DIR)
    if not os.path.exists(run.IMAGES_DIR):
        os.makedirs(run.IMAGES_DIR)
        
    # Disable WeChat push for testing
    run.send_wechat_markdown = lambda x: print("[Test] Skipped WeChat Markdown Push")
    run.send_wechat_image = lambda x: print("[Test] Skipped WeChat Image Push")
    
    # Mock upload_file to use inline data for testing
    class MockFile(dict):
        def __init__(self, data_dict):
            super().__init__(data_dict)
            self.state = type('obj', (object,), {'name': 'ACTIVE'})
            self.name = "mock_name"
    
    def mock_upload(path):
        print("[Test] Using inline data instead of upload_file to bypass proxy issues.")
        return MockFile({"mime_type": "application/pdf", "data": open(path, "rb").read()})
        
    run.genai.upload_file = mock_upload
    run.genai.get_file = lambda name: MockFile({"mime_type": "application/pdf", "data": b""})
    run.genai.delete_file = lambda name: None

    analyze_papers_with_gemini(papers)
else:
    print("Failed to find the paper.")
