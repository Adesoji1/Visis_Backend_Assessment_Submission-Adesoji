# import requests
# import pdf

# url1 = "http://127.0.0.1:5000/generate-summary"
# url = "http://127.0.0.1:5000//generate-points"


# def call_summary(summary):
#     data = {"summarize": summary}
#     res = requests.post(url1, json=data)

#     if res.status_code == 200:
#         print(res)
#         # summary = res.json()

#         return summary
#     else:
#         print(f"Request failed with status code {res.status_code}")
#         return None


# print(call_summary(pdf.readPdf("CoverApp.pdf")))


# def call_points(points):
#     data = {"points": points}
#     res = requests.post(url, json=data)

#     if res.status_code == 200:
#         summary = res.json()

#         return summary
#     else:
#         print(f"Request failed with status code {res.status_code}")
#         return None


# testPdf = "Cover Letter Cash App.pdf"
# 
# print(call_points(pdf.readPdf("CoverApp.pdf")))

import requests
import pdf

url1 = "http://127.0.0.1:5000/generate-summary"
url = "http://127.0.0.1:5000/generate-points"


def call_summary(summary):
    data = {"summarize": summary} 
    # print(data)
    res = requests.post(url1, json=data)
    # print(res)
    # print(f"Response status code: {res.status_code}")
    # print(f"Response content: {res.text}")

    if res.status_code == 200:
        try:
            summary = res.text
            return summary
        except requests.exceptions.JSONDecodeError:
            print("Failed to decode JSON response")
            return None
    else:
        print(f"Request failed with status code {res.status_code}")
        return None


print(call_summary(pdf.readPdf("Cover.pdf")))



def call_points(points):
    data = {"points": points}
    res = requests.post(url, json=data)

    if res.status_code == 200:
        summary = res.text

        return summary
    else:
        print(f"Request failed with status code {res.status_code}")
        return None





# print(call_points(pdf.readPdf("Profile-4.pdf")))