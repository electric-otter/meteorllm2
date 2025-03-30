import requests

def search_duckduckgo(query):
    # Remove extra spaces by splitting and joining words with a single space
    query = "+".join(query.split())
    url = f"https://api.duckduckgo.com/?q={query}&format=json&t=h_"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if "RelatedTopics" in data:
            results = [topic["Text"] for topic in data["RelatedTopics"] if "Text" in topic]
            return results[:5]  # Limit to top 5 results
    return ["Hmm, I couldn't find anything useful. Maybe try a different wording? 🤔"]

def chatbot_response(user_input):
    # Remove leading/trailing spaces and convert to lowercase
    user_input = user_input.strip().lower()

    # Try searching for something using DuckDuckGo
    results = search_duckduckgo(user_input)
    if results:
        return "Here’s what I found for you! 🧐\n" + "\n".join(results)
    
    # Default response for unrecognized input
    return "Not sure what you mean. Try asking me something! 💡"

# Chatbot loop
print("AI Chatbot (type 'exit' to stop)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Alright, see you next time! 👋 Take care!")
        break
    response = chatbot_response(user_input)
    print("AI:", response)
