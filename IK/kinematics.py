import math
from typing import Tuple

class EezybotKinematics:
    def __init__(
        self,
        L1: float = 9.5,   # Base to shoulder pivot (cm)
        L2: float = 13.5,  # Lower arm length (cm)
        L3: float = 14.7,  # Forearm length (cm)
        L4: float = 9.0,   # End-effector claw offset (cm)
        # Calibration offsets and ratios
        base_ratio: float = 2.0,
        base_offset: float = 90.0,
        shoulder_offset: float = 0.0,
        elbow_offset: float = 90.0,
        min_z: float = 0.0, # Ground collision threshold
        x_offset: float = 46.0,  # Offset for x-axis
    ):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        
        self.base_ratio = base_ratio
        self.base_offset = base_offset
        self.shoulder_offset = shoulder_offset
        self.elbow_offset = elbow_offset
        self.min_z = min_z
        self.x_offset = x_offset

    def is_reachable(self, x: float, y: float, z: float) -> Tuple[bool, str]:
        """Checks if a target coordinate (x, y, z) in cm is reachable."""
        x = self.x_offset - x  # Adjust for x-axis offset

        if z < self.min_z:
            return False, f"Collision: Target below ground (z={z:.2f}cm)"
            
        xy_dist = math.hypot(x, y)
        if xy_dist <= self.L4:
            return False, f"Target inside base radius: distance {xy_dist:.2f}cm <= {self.L4}cm"

        r = xy_dist - self.L4
        dz = z - self.L1
        d = math.hypot(r, dz)

        if d < 1e-6:
            return False, "Target coincides directly with shoulder pivot"

        max_reach = self.L2 + self.L3
        min_reach = abs(self.L2 - self.L3)

        if d > max_reach:
            return False, f"Target out of reach: distance {d:.2f}cm > max {max_reach:.2f}cm"
        if d < min_reach:
            return False, f"Target too close: distance {d:.2f}cm < min {min_reach:.2f}cm"
            
        return True, "OK"

    def solve_ik(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Computes joint angles in radians. 
        theta2 is absolute from horizontal, theta3 is interior angle (0 = straight).
        """
        reachable, msg = self.is_reachable(x, y, z)
        if not reachable:
            raise ValueError(msg)

        x = self.x_offset - x  # Adjust for x-axis offset

        # 1. Base angle (yaw)
        theta1 = math.atan2(y, x)

        # 2. Planar projection
        r = math.hypot(x, y) - self.L4
        dz = z - self.L1
        d_sq = r**2 + dz**2
        d = math.sqrt(d_sq)

        # 3. Shoulder angle (theta2) - absolute from horizontal
        theta = math.atan2(dz, r)
        cos_beta = (self.L2**2 - self.L3**2 + d_sq) / (2 * self.L2 * d)
        beta = math.acos(max(-1.0, min(1.0, cos_beta)))
        theta2 = theta + beta

        # 4. Forearm angle (theta3) - interior angle relative to L2
        cos_theta3 = (d_sq - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        theta3 = math.acos(max(-1.0, min(1.0, cos_theta3)))

        return theta1, theta2, theta3
        
    def solve_fk(self, theta1: float, theta2: float, theta3: float) -> Tuple[float, float, float]:
        """
        Forward kinematics for round-trip validation.
        theta3=0 is a straight arm. For the elbow-up configuration, 
        the absolute angle is theta2 - theta3.
        """
        absolute_elbow_angle = theta2 - theta3
        
        r = self.L2 * math.cos(theta2) + self.L3 * math.cos(absolute_elbow_angle)
        dz = self.L2 * math.sin(theta2) + self.L3 * math.sin(absolute_elbow_angle)
        
        z = dz + self.L1
        total_r = r + self.L4
        
        x = total_r * math.cos(theta1)
        y = total_r * math.sin(theta1)

        x = self.x_offset - x  # Adjust for x-axis offset
        return x, y, z

    def joint_angles_to_servo_degrees(
        self, theta1: float, theta2: float, theta3: float
    ) -> Tuple[int, int, int]:
        """
        Converts joint angles (radians) into physical servo angles.
        Returns clamped angles bounded to [0, 180].
        """
        t1_deg = math.degrees(theta1)
        t2_deg = math.degrees(theta2)
        t3_deg = math.degrees(theta3)

        raw_m1 = self.base_offset - (self.base_ratio * t1_deg)
        raw_m2 = t2_deg + self.shoulder_offset
        raw_m3 = self.elbow_offset - t3_deg + t2_deg

        def clamp_servo(val: float, name: str) -> int:
            clamped = max(0.0, min(180.0, val))
            if clamped != val:
                print(f"Warning: {name} servo angle {val:.1f} clamped to {clamped:.0f}")
            return int(round(clamped))

        return (
            clamp_servo(raw_m1, "Base"),
            clamp_servo(raw_m2, "Shoulder"),
            clamp_servo(raw_m3, "Elbow")
        )