"""UR5e 목표 자세: 각 관절의 홈 자세 기준 오프셋(rad).

값을 수정한 뒤 실행 프로그램을 다시 시작하면 FK, Jacobian, 뷰어에 적용됩니다.
0.0은 홈 자세를 유지한다는 뜻이며, 절대 관절각 0을 뜻하지 않습니다.
"""

JOINT_OFFSETS: dict[str, float] = {
    "shoulder_pan_joint": 0.9,
    "shoulder_lift_joint": 0.5,
    "elbow_joint": -1.1,
    "wrist_1_joint": 1.3,
    "wrist_2_joint": -0.8,
    "wrist_3_joint": 2.1
}
