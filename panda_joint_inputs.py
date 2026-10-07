"""Panda 목표 자세: 각 관절의 홈 자세 기준 오프셋(rad).

값을 수정한 뒤 실행 프로그램을 다시 시작하면 FK, Jacobian, 뷰어에 적용됩니다.
목표 관절각 = 홈 관절각 + JOINT_OFFSETS.
"""

JOINT_OFFSETS: dict[str, float] = {
    "joint1": 1.5,
    "joint2": -0.90,
    "joint3": 1.1,
    "joint4": -0.80,
    "joint5": 0.85,
    "joint6": 1.3,
    "joint7": -0.50,
}
