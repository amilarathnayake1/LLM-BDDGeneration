import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def setup_nltk():
    """Download required NLTK data"""
    try:
        print("Downloading NLTK data...")
        print("1. Downloading wordnet...")
        nltk.download('wordnet')
        print("2. Downloading punkt...")
        nltk.download('punkt')
        print("3. Downloading omw-1.4...")
        nltk.download('omw-1.4')
        print("\nSuccessfully downloaded all NLTK data!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    setup_nltk()