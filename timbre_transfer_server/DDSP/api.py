from pyngrok import ngrok, conf

conf.get_default().auth_token = "2ycqdvzNP6OqqVrPAFmdaSHJNOZ_2oBWcyhQxHZQ23xVuTMZ8"
# Create tunnel for port 8080
public_url = ngrok.connect(8080)
print("Public API URL:", public_url)
input("Press Enter to stop...")
