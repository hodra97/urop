"""TidyBot의 home 기준 FK 오프셋: 직선관절은 m, 회전관절은 rad.

베이스 3축과 팔 7축에 작은 예제 이동을 적용합니다.
목표 관절값 = home 관절값 + JOINT_OFFSETS. 0.0은 home 유지입니다.
이 입력은 FK용이며, IK는 베이스를 포함한 10관절을 함께 사용합니다.
"""

JOINT_OFFSETS: dict[str, float] = {
    "joint_x": 0.80,   # 베이스 x 이동 (m)
    "joint_y": -0.60,  # 베이스 y 이동 (m)
    "joint_th": 1.3,  # 베이스 yaw 회전 (rad)
    "joint_1": 0.95,
    "joint_2": -0.60,
    "joint_3": 1.5,
    "joint_4": 0.90,
    "joint_5": -1.3,
    "joint_6": 0.89,
    "joint_7": 0.8,
}
