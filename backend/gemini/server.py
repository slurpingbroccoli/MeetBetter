from flask import Flask, request, jsonify
from scrape import scrape_linkedin_profile

# ----------------- Config -----------------
SECRET_KEY = "extremelyrarepictureofaseaspugnar"
PORT = 8000
# -----------------------------------------

app = Flask(__name__)

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.json
    if not data or data.get("key") != SECRET_KEY:
        return jsonify({"error": "unauthorized"}), 403

    url = data.get("url")
    if not url:
        return jsonify({"error": "missing 'url' parameter"}), 400

    try:
        profile_data = scrape_linkedin_profile(url)
        return jsonify(profile_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"LinkedIn Scraper API running on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT)