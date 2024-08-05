from flask import Flask, request, jsonify
import models  # Ensure models.py is correctly implemented and imported
import pdf  # Ensure pdf.py is correctly implemented and imported

app = Flask(__name__)

@app.route("/pred")
def user():
    return "I like"

@app.route("/generate-summary", methods=["POST"])
def create1_user():
    try:
        data = request.get_json()
        # print(data)
        summary_data = data.get("summarize")
        # print(summary_data)
        if not summary_data:
            return jsonify({"error": "No text provided for summarization"}), 400

        summary_model = models.generate_book_Summary(summary_data)
        print(summary_model)
        return jsonify(summary_model)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/generate-points", methods=["POST"])
def create_user():
    try:
        data = request.get_json()
        points_data = data.get("points")

        if not points_data:
            return jsonify({"error": "No text provided for points generation"}), 400

        points_model = models.generate_book_Points(points_data)

        return jsonify(points_model)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)



# from flask import Flask, request
# import urllib.error
# import models
# import pdf

# app = Flask(__name__)

# @app.route("/pred")
# def user():
#     return "I like"

# @app.route("/generate-summary", methods=["POST"])
# def generate_summary():
#     try:
#         data = request.get_json()
#         summaryData = data["summarize"]
#         summaryModel = models.generate_book_Summary(summaryData)

#         return summaryModel

#     except Exception as e:
#         return {"error": "An unexpected error occurred: The link exceeds max limit of text" + str(e)}, 500

# @app.route("/generate-points", methods=["POST"])
# def generate_points():
#     try:
#         data = request.get_json()
#         summaryData = data["points"]
#         points = models.generate_book_Points(summaryData)

#         return points

#     except Exception as e:
#         return {"error": "An unexpected error occurred: The link exceeds max limit of text" + str(e)}, 500

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)
