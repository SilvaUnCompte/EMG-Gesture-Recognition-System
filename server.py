from flask import Flask, request, jsonify # type: ignore

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    # Récupération des données envoyées au format JSON
    data = request.get_json()

    if not data or "values" not in data:
        return jsonify({"error": "Need json with key 'values'"}), 400

    values = data["values"]

    if not isinstance(values, list) or len(values) != 8:
        return jsonify({"error": "Need exactly 8 values in the list"}), 400

    # Example processing: sum and average
    total = sum(values)
    average = total / len(values)

    result = {
        "sum": total,
        "average": average
    }

    # Affichage en console
    print("Values :", values)
    print("RResult :", result)

    # Retourne aussi le résultat au client
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
