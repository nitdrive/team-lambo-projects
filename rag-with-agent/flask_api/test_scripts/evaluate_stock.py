import json

if __name__ == '__main__':
    with open(file="nvidia.json") as f:
        nvidia_details = json.load(f)
        for key in nvidia_details:
            print(f"\"{key}\",")

