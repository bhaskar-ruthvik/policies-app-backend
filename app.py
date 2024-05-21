from flask import Flask,jsonify,request
from utils import getCategoryOfInput,getResponseFromLLM, formatParagraphType,formatFlowchartType

app = Flask(__name__)
model = "gpt-4-0125-preview"

@app.route('/',methods=["POST"])
def index():
    if request.method == "POST": 
        ip = request.form.get("body")
        cat = getCategoryOfInput(model,ip)
        content = getResponseFromLLM(model,ip,cat)
        if cat=="Informative Paragraph Question":
            headings, slugs = formatParagraphType(content)
            body = {
                    "headings": headings,
                    "slugs": slugs
                }
        elif cat=="Procedure-Based Question":
            body = formatFlowchartType(content)
        else:
             [val, cont] = content.split("\n\n")
             body = {
                 "value": val,
                 "content": cont
             }
        data = {
                "type": cat,
                "body": body
            }
        return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
