# A Study on vulnerability of recognition area for Cloud-based autonomous vehicles using Light-weight Seam Carving
### 본 연구는 자율주행 차량의 실시간 영상을 해석하고 처리하는 개체 인식 서비스를 공격하는 알고리즘을 구현하고자 한다. 
### Tiny-YOLO를 통해 인식된 차량과 도로 개체의 영역을 개선된 Seam Carving을 통해 실시간으로 제거하는 방식이다.

## Author: Tackhyun Jung

## Status: 완료

![1](https://user-images.githubusercontent.com/41291493/109094841-fcb02c00-775d-11eb-89dc-61d885cef0ea.png)
![2](https://user-images.githubusercontent.com/41291493/109094851-ff128600-775d-11eb-85fb-b75b69bba58b.png)
![3](https://user-images.githubusercontent.com/41291493/109094853-0043b300-775e-11eb-84e4-f392c7081893.png)

### 핵심목표
1) Tiny-YOLO를 통해 인식된 차량과 도로 개체의 영역의 인식 알고리즘 구현 (완료)
2) 개선된 Seam Carving을 통해 실시간으로 차량과 도로 개체 제거 알고리즘 구현 (완료)
3) 실시간 스트리밍 환경에 적용하였을때의 처리시간 측정 알고리즘 구현 (완료)

---

### 사용된 기술
* Tiny-YOLO
* Seam Carving
* Adversarial Ataack

---

### Requirement
* Tensorflow 3x
* cv2
* numpy
* tensorflow

---

### Usage

```
> python {script}.py
```

---
