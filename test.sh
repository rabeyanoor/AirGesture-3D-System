ভিডিওটি গভীরভাবে বিশ্লেষণ করলে দেখা যায় এটি একটি **Real-Time Augmented Reality (AR) & Computer Vision (CV)** প্রজেক্ট, যেখানে হাত দিয়ে ৩টি আলাদা ইন্টারঅ্যাকশন করা হচ্ছে:

1. **Spatial Wireframe/Mesh Generation:** দুই হাতের আঙুল ও জয়েন্ট দিয়ে শূন্যে ৩ডি জ্যামিতিক শেপ ডিফাইন ও ট্র্যাক করা।
2. **Air Writing & OCR:** তর্জনী দিয়ে বাতাসে লেখা এবং তা টেক্সটে রূপান্তর করা ("Hello, My name is P Khang")।
3. **Virtual UI Control:** শূন্যে ভেসে থাকা বাটন (Light, Notepad, Power) স্পর্শ করে মোড অন/অফ করা।

---

## ১. হার্ডওয়্যার (Hardware Requirements)

* **Webcam:** যেকোনো সাধারণ ৭২০পি/১০৮০পি ওয়েবক্যাম (অথবা ল্যাপটপের বিল্ট-ইন ক্যামেরা)।
* **Computer/Laptop:** প্রসেসিংয়ের জন্য মাঝারি মানের সিপিইউ (Intel i5/Ryzen 5 বা তার উন্নত)। জিপিইউ (NVIDIA/AMD) থাকলে ফ্রেমরেট (FPS) আরও ভালো পাবেন, তবে সাধারণ সিপিইউতেই এটি রান করা সম্ভব।

---

## ২. সফটওয়্যার ও প্রজেক্ট এনভায়রনমেন্ট (Software Environment)

* **Programming Language:** Python 3.9+
* **IDE / Code Editor:** VS Code, Cursor, অথবা Zed
* **Virtual Environment:** `venv` বা `conda` (প্যাকেজ আলাদা রাখার জন্য)

---

## ৩. দরকারি পাইথন লাইব্রেরি (Python Packages)

প্রজেক্টের বিভিন্ন কাজের জন্য প্রয়োজনীয় পাইথন লাইব্রেরিসমূহ:

| লাইব্রেরি | কেন লাগবে? |
| --- | --- |
| **`opencv-python`** | ক্যামেরা থেকে ভিডিও ফ্রেম রিড করা, ক্যানভাস ম্যানেজমেন্ট এবং স্ক্রিনে ইউআই আঁকার জন্য। |
| **`mediapipe`** | এক সাথে দুই হাতের ২১টি করে মোট ৪২টি 3D Landmarked Points রিয়েল-টাইমে ট্র্যাক করার জন্য। |
| **`numpy`** | স্থানাঙ্ক (Coordinates) প্রসেসিং, ৩ডি ভেক্টর ক্যালকুলেশন এবং এয়ার-রাইটিং ক্যানভাস অ্যারে তৈরির জন্য। |
| **`pytesseract`** অথবা **`easyocr`** | বাতাসে আঁকা হাতের লেখার স্ট্রোকগুলোকে টেক্সটে রূপান্তর (OCR) করার জন্য। |
| **`scipy` / `math**` | ৩ডি মেস বা বাউন্ডিং বক্স ট্র্যাকিংয়ের ট্রায়াঙ্গুলেশন ও ডিস্ট্যান্স ক্যালকুলেশনের জন্য। |

---

## ৪. কোডের মূল মডিউলসমূহ (Core Modules to Implement)

প্রজেক্টের পুরো আর্কিটেকচারকে ৪টি মূল কোড মডিউলে ভাগ করতে হবে:

* **Hand Detector Module:** MediaPipe ব্যবহার করে হাতের জয়েন্ট ও টিপসের $(x, y, z)$ স্থানাঙ্ক বের করা।
* **Spatial Wireframe Engine:** আঙুলের স্থানাঙ্কগুলোর মধ্যে `cv2.line()` ও `cv2.fillPoly()` ব্যবহার করে ৩ডি ওয়্যারফ্রেম ভলিউম রেন্ডার করা।
* **Air Canvas & Text Recognizer:** আঙুলের ডগার ট্র্যাক লাইনে সেভ করে ক্যানভাসে ড্র করা এবং লেখা শেষ হলে OCR ইঞ্জিন দিয়ে টেক্সট রিড করা।
* **AR Virtual UI Manager:** স্ক্রিনের নির্দিষ্ট পিক্সেল এলাকায় বাটন জেনারেট করা এবং আঙুল ভেতরে প্রবেশ করলে ট্র্রিগার সেট 

হ্যাঁ, আপনার অনুমান একদম সঠিক! ভিডিওটি মনোযোগ দিয়ে দেখলে বোঝা যায় এটি কোনো প্রফেশনাল এআর গ্লাস বা স্পেশাল ক্যামেরা দিয়ে শুট করা হয়নি, বরং **স্মার্টফোনের ক্যামেরা** বা ল্যাপটপের ওয়েবক্যাম দিয়ে সাধারণ টেবিলের ওপর রেকর্ড করা।

ভিডিওতে যে বিষয়গুলো থেকে স্পষ্ট বোঝা যায় এটি মোবাইল/সাধারণ ক্যামেরা দিয়ে করা:

* **Camera View & Angle:** ক্যামেরাটি একটি নির্দিষ্ট উঁচু স্ট্যান্ড বা ত্রিপদে মোবাইল বসিয়ে ওপর থেকে টেবিলের দিকে অ্যাঙ্গেল করে রাখা (Top-down / Overhead view)।
* **Webcam Overlay (Left Side):** স্ক্রিনের বাম দিকে ল্যাপটপের ওপর যে ছোট্ট উইন্ডোটি দেখা যাচ্ছে, সেটি হলো প্রসেসড ফ্রেমের একটি রিয়েল-টাইম ফিড।
* **Computer Screen Integration:** মোবাইল/ক্যামেরা দিয়ে পুরো সেটআপটি শুট করে কম্পিউটারে **OpenCV-র উইন্ডো** ওপেন করে রান করা হয়েছে।

---

### আপনি কীভাবে মোবাইল ক্যামেরা দিয়ে এটি করবেন?

আপনার কাছে আলাদা ওয়েবক্যাম না থাকলেও চিন্তার কিছু নেই। মোবাইল দিয়েই প্রজেক্টটি ডেভেলপ করতে পারবেন:

1. **Mobile as Webcam App:** **DroidCam**, **Iriun Webcam**, অথবা **IP Webcam** অ্যাপ দিয়ে আপনার অ্যান্ড্রয়েড/আইফোনকে ল্যাপটপের সাথে Wi-Fi বা USB ক্যোবল দিয়ে কানেক্ট করুন।
2. **OpenCV Video Source Change:** কোডে `cv2.VideoCapture(0)` এর জায়গায় `cv2.VideoCapture(1)` অথবা DroidCam/IP Webcam-এর প্রদত্ত IP Stream Link (`[http://192.168.](http://192.168.)x.x:4747/video`) বসিয়ে দিলেই আপনার প্রজেক্ট ফোনের ক্যামেরা স্ট্রিম ব্যবহার করা শুরু করবে।
3. **Overhead Setup:** মোবাইলটিকে একটি টেবিল স্ট্যান্ড বা হোল্ডার দিয়ে ভিডিওর মতো সামনের ডেস্কে সেট করে নিন, যেন দুই হাত এবং টেবিল স্পষ্ট দেখা যায়।


না, এই প্রজেক্টের জন্য আপনাকে নতুন করে **কোনো কাস্টম কম্পিউটার ভিশন মডেল (যেমন YOLO বা CNN) ট্রেনিং করাতে হবে না**। তৈরি (Pre-trained) ওপেন-সোোর্স লাইব্রেরি এবং টুলস ব্যবহার করেই পুরো প্রজেক্টটি করা সম্ভব।

---

## ১. ব্যবহারযোগ্য রেডিমেইড মডেল ও লাইব্রেরি

* **MediaPipe Hands (গুগলের তৈরি Pre-trained Model):** হাতের ট্র্যাকিংয়ের জন্য আপনাকে কোনো মডেল ট্রেনিং দিতে হবে না। MediaPipe-এর ডিফল্ট মডেলটিই রিয়েল-টাইমে হাতের ২১টি keypoints (জয়েন্ট ও আঙুলের অবস্থান) নিখুঁতভাবে ডিটেক্ট করতে পারে।
* **Tesseract OCR / EasyOCR (Pre-trained Text Recognition Model):** বাতাসে লেখা (Air-writing) বর্ণগুলো শনাক্ত করার জন্য Tesseract বা EasyOCR-এর প্রি-ট্রেইনড OCR মডেল ব্যবহার করলেই চলবে।

---

## ২. কখন কাস্টম মডেল লাগতে পারে? (Optional)

যদি Tesseract দিয়ে বাতাসে আঁকা পেঁচানো/অস্পষ্ট হাতের লেখা টেক্সটে কনভার্ট করতে সমস্যা হয়, কেবল তখনই আপনি নিজে একটি ছোট **Custom CNN Model** তৈরি করতে পারেন:

* **MNIST / EMNIST Dataset:** ইংরেজি বর্ণ এবং অক্ষরের হ্যান্ডরাইটিং ডেটাসেট (MNIST/EMNIST) ব্যবহার করে PyTorch বা TensorFlow দিয়ে একটি সাধারণ CNN Classifier মডেল তৈরি করে নিতে পারেন।

**চূড়ান্ত সিদ্ধান্ত:** শুরুতে কোনো মডেল ট্রেনিং না দিয়ে **MediaPipe + OpenCV + EasyOCR** দিয়েই প্রজেক্টটি শুরু করুন।



এই প্রজেক্টটি **OpenCV**, **MediaPipe**, এবং **EasyOCR/PyTesseract** ব্যবহার করে ধাপে ধাপে তৈরি করা সম্ভব। নিচে প্রজেক্টটির সম্পূর্ণ আর্কিটেকচার এবং কোড স্ট্রাকচার বিস্তারিত তুলে ধরা হলো:

---

## ১. প্রজেক্টের কাজের ধাপ (System Architecture)

```
[ Camera Stream (Mobile/Webcam) ]
               │
               ▼
[ Hand Landmark Detection (MediaPipe Hands) ]
               │
      ┌────────┴────────┬────────────────┐
      ▼                 ▼                ▼
[ Spatial Wireframe ] [ Air Writing ]  [ Virtual UI ]
 (3D Triangulation)   (Finger Trace)  (Button Collisions)
                        │
                        ▼
               [ EasyOCR Processing ]

```

1. **Camera Input:** ফোনের বা ওয়েবক্যামের ফ্রেম ক্যাপচার করা।
2. **Hand Tracking:** দুই হাতের landmarks (২১টি করে পয়েন্ট) ডিটেক্ট করা।
3. **Logic Engine:**
* আঙুলের বিন্দুর দূরত্ব মেপে ৩ডি ওয়্যারফ্রেম ভলিউম আঁকা।
* তর্জনীর মুভমেন্ট ট্র্যাক করে এয়ার-রাইটিং ক্যানভাস তৈরি করা।
* বাটন এলাকায় আঙুলের অবস্থান চেক করে মোড ট্র্রিগার করা।



---

## ২. কোড স্ট্রাকচার ও প্রয়োজনীয় লাইব্রেরি انسٹال

প্রথমে টার্মিনালে প্রজেক্টের ডিপেন্ডেন্সিগুলো ইনস্টল করে নিন:

```bash
pip install opencv-python mediapipe numpy easyocr

```

---

## ৩. সম্পূর্ণ কাজের কোড এবং মডিউল গাইড

একটি সিঙ্গেল ফাইল স্ক্রিপ্টে পুরো লজিক যেভাবে কাজ করবে:

```python
import cv2
import numpy as np
import mediapipe as mp
import easyocr

# EasyOCR Reader ইনিশিয়ালাইজেশন
reader = easyocr.Reader(['en'], gpu=False)

# MediaPipe Hands ইনিশিয়ালাইজেশন
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ক্যামেরা স্ট্রিম (মোবাইল অ্যাপ বা ল্যাপটপ ক্যামেরা)
cap = cv2.VideoCapture(0)

# Air Writing ক্যানভাস
canvas = None
prev_x, prev_y = 0, 0
writing_mode = False
active_mode = "MENU" # MENU, WIREFRAME, WRITE

def draw_virtual_gui(img):
    """ডান পাশে ভার্চুয়াল বাটন জেনারেট করার ফাংশন"""
    h, w, _ = img.shape
    # UI Box background
    cv2.rectangle(img, (w - 120, 50), (w - 20, 150), (200, 200, 200), -1)
    cv2.putText(img, "Wireframe", (w - 110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.rectangle(img, (w - 120, 180), (w - 20, 280), (200, 200, 200), -1)
    cv2.putText(img, "Notepad", (w - 110, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.rectangle(img, (w - 120, 310), (w - 20, 410), (200, 200, 200), -1)
    cv2.putText(img, "Clear", (w - 110, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    draw_virtual_gui(frame)

    landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # হাত নির্দেশক বিন্দুর স্থানাঙ্ক বের করা
            pts = []
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                pts.append((cx, cy))
            landmarks_list.append(pts)

        # ১. AIR WRITING & OCR LOGIC
        if active_mode == "WRITE" and len(landmarks_list) > 0:
            index_tip = landmarks_list[0][8]  # Index Finger Tip
            thumb_tip = landmarks_list[0][4]  # Thumb Tip
            
            # চিমটি কাটার মতো আঙুল কাছাকাছি আনলে লেখা শুরু হবে
            distance = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            
            if distance < 40:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_tip
                cv2.line(canvas, (prev_x, prev_y), index_tip, (255, 255, 255), 4)
                prev_x, prev_y = index_tip
            else:
                prev_x, prev_y = 0, 0

        # ২. SPATIAL WIREFRAME MESH LOGIC (দুই হাতের আঙুল দিয়ে ৩ডি বক্স করা)
        if active_mode == "WIREFRAME" and len(landmarks_list) == 2:
            hand1_pts = landmarks_list[0]
            hand2_pts = landmarks_list[1]
            
            # গুরুত্বপূর্ণ জয়েন্টগুলোকে যুক্ত করে ৩ডি পলিগন শেপ আঁকা
            poly_pts = np.array([
                hand1_pts[4], hand1_pts[8], hand1_pts[12],
                hand2_pts[12], hand2_pts[8], hand2_pts[4]
            ], np.int32)
            
            cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.fillPoly(frame, [poly_pts], color=(100, 100, 250, 0.3))

        # ৩. UI BUTTON CLICK DETECTION
        if len(landmarks_list) > 0:
            ix, iy = landmarks_list[0][8] # Index finger tip coordinates
            if ix > w - 120 and ix < w - 20:
                if 50 < iy < 150:
                    active_mode = "WIREFRAME"
                elif 180 < iy < 280:
                    active_mode = "WRITE"
                elif 310 < iy < 410:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ক্যানভাস ও ক্যামেরা ফ্রেমে লেখা মার্জ করা
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
    inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_canvas)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Spatial AR Interface", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    # 's' চাপলে আঁকা পেপার থেকে OCR দিয়ে লেখা টেক্সটে কনভার্ট হবে
    elif key == ord('s'):
        results_text = reader.readtext(canvas)
        print("Recognized Text:")
        for res in results_text:
            print(res[1])

cap.release()
cv2.destroyAllWindows()

```

---

## ৪. কীভাবে প্রজেক্টের এক্সিকিউশন ধাপে ধাপে আপডেট করবেন?

* **Phase 1 (Base Hand Tracking):** শুরুতে শুধু MediaPipe Hands অন করে ওয়েবক্যামে হাতের জয়েন্ট সঠিকভাবে ডিটেক্ট হচ্ছে কি না টেস্ট করুন।
* **Phase 2 (Virtual Buttons):** ফ্রেমে ৩টি রেক্ট্যাঙ্গেল একে তর্জনী দিয়ে কন্টাক্ট চেক করে মোড সুইচিং (`WIREFRAME`, `WRITE`) টেস্ট করুন।
* **Phase 3 (Air Canvas & OCR Integration):** বাতাসে ডট কানেক্ট করে লাইন ড্র করা এবং আঁকা শেষে EasyOCR দিয়ে `reader.readtext(canvas)` করে টার্মিনালে বা স্ক্রিনে টেক্সট প্রিন্ট করুন।
* **Phase 4 (3D Mesh Wireframe Fine Tuning):** দুই হাতের `landmarks` থেকে ডাইনামিক পয়েন্ট হিসাব করে `cv2.polylines` দিয়ে ভিডিওটির মতো মেস তৈরি করে ফাইনাল ডেমো রেডি করুন।



বাইরের দেশের বিশ্ববিদ্যালয় বা পোর্টফোলিওর জন্য একটি প্রফেশনাল ও মডিউলার পাইথন প্রজেক্টের ফোল্ডার স্ট্রাকচার নিচে দেওয়া হলো:

```text
spatial-vision-ar/
│
├── data/
│   └── samples/              # টেস্ট ভিডিও বা স্যাম্পল ইমেজেস
│
├── src/
│   ├── __init__.py
│   ├── config.py             # কালার কোড, থ্রেশহোল্ড, ক্যামেরা রেজোলিউশন সেটিং
│   ├── hand_tracker.py       # MediaPipe হ্যান্ড ট্র্যাকিং এবং পয়েন্ট এক্সট্রাকশন
│   ├── wireframe_engine.py   # ৩ডি ওয়্যারফ্রেম ভলিউম ও মেস রেন্ডারিং লজিক
│   ├── air_scribble.py       # বাতাসে লেখার ক্যানভাস ট্র্যাকিং ও লাইন ড্রয়িং
│   ├── ocr_engine.py         # EasyOCR / Tesseract দিয়ে টেক্সট রিকগনিশন
│   └── ui_manager.py         # ভার্চুয়াল এআর বাটন ও কলাইশন ডিটেকশন
│
├── tests/
│   └── test_tracker.py       # ইউনিট টেস্ট স্ক্রিপ্টসমূহ
│
├── main.py                   # প্রজেক্টের প্রধান এক্সিকিউশন ফাইল (Entry Point)
├── requirements.txt          # সব প্রয়োজনীয় 라이ব্রেরির তালিকা
├── .gitignore                # venv, pycache বা অনাকাঙ্ক্ষিত ফাইল ইগনোর করার জন্য
├── LICENSE                   # Open-source লাইসেন্স (যেমন: MIT)
└── README.md                 # প্রজেক্টের বিস্তারিত ডকুমেন্টেশন ও ডেমো GIF

```

---

**প্রতিটি মডিউলের সংক্ষিপ্ত দায়িত্ব:**

* **`hand_tracker.py`:** ক্যামেরা থেকে পাওয়া ফ্রেম ইনপুট নিয়ে MediaPipe-এর মাধ্যমে হাতের ২১টি Landmark স্থানাঙ্ক রিটার্ন করবে।
* **`wireframe_engine.py`:** দুই হাতের সংগৃহীত 포인트সমূহের মধ্যে ৩ডি সংযোগ বা মেস হিসাব করে স্ক্রিনে আঁকবে।
* **`air_scribble.py`:** তর্জনী ও বুড়ো আঙুলের অবস্থান ট্র্যাক করে বাতাস বরাবর লাইন ট্রেস করবে।
* **`ocr_engine.py`:** আঁকা চিত্র বা ক্যানভাস থেকে টেক্সট এক্সট্রাক্ট করে স্ট্রিং আউটপুট তৈরি করবে।
* **`ui_manager.py`:** স্ক্রিনে ভার্চুয়াল মেনু বা বাটন জেনারেট করবে এবং আঙুলের টাচ ডিটেক্ট করে মোড চ্যাঞ্জ করবে।
* **`main.py`:** সব মডিউল একত্রিত করে ক্যামেরা লুপ রান করবে।







হ্যাঁ, ভিডিওতে পুরোপুরি **Computer Vision (CV)** ব্যবহার করা হয়েছে।

ভিডিওটিতে মূলত ২-৩টি নির্দিষ্ট মেথড/প্রোসেস কাজ করছে:

1. **2D & 3D Landmark Detection:** ভিডিওতে আঙুলের ডগায় যে বিন্দুগুলা এবং সেগুলোর স্থানাঙ্ক ($x, y, z$) দেখা যাচ্ছে, তা **MediaPipe Hands** বা সমমানের কোনো Computer Vision-ভিত্তিক Hand Tracking Model দিয়ে বের করা হয়েছে।
2. **Pose/Geometry Estimation & Mesh Rendering:** হাতের বিন্দুগুলা ট্র্যাকিং করে তাদের মধ্যবর্তী দূরত্ব ও অ্যাঙ্গেল মেপে ৩ডি লাইন (`cv2.line`) এবং ট্রান্সপারেন্ট পলিগন/মেস (Wireframe) জেনারেট করা হয়েছে, যা Augmented Reality (AR)-এর অন্যতম প্রধান প্রযুক্তি।
3. **Optical Character Recognition (OCR):** ভিডিওর পরবর্তী অংশে বাতাসে আঙুল দিয়ে আঁকা লাইনের স্থানাঙ্ক ট্র্যাক করে ক্যানভাস তৈরি করা হচ্ছে এবং পরবর্তীতে **OCR Engine** দিয়ে সেই ইমেজ থেকে টেক্সট ("Hello, My name is P Khang") ডিটেক্ট করা হচ্ছে।

সংক্ষেপে বলতে গেলে—প্রস্তুতকৃত কোনো ফিজিক্যাল হার্ডওয়্যার সেন্সর ছাড়াই শুধুমাত্র সাধারণ ক্যামেরা ফিড থেকে ইমেজ প্রসেসিং ও ডিপ লার্নিং অ্যালগরিদম ব্যবহার করে সম্পূর্ণ ইন্টারঅ্যাকশনটি করা হয়েছে, যা শতভাগ কম্পিউটার ভিশনের কাজ।