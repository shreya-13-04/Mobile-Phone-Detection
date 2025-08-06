file_path = r"C:\Users\Shreya\Desktop\EvolveTag\phone_detection\coco.names"

try:
    with open(file_path, "r") as f:
        print("File found! Here are the first 5 lines:")
        print("\n".join(f.readlines()[:5]))
except FileNotFoundError:
    print("Error: File not found!")
except Exception as e:
    print(f"Unexpected error: {e}")
