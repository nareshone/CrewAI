import pickle
try:
    with open('conversation_memory.pkl', 'rb') as f:
        data = pickle.load(f)
        print(type(data))
        print(data)
except Exception as e:
    print("Error reading pickle file:", e)