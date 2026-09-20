import re
import serial
from enum import Enum
import time
import math
from kinematics import EezybotKinematics

class ServoName(Enum):
    SERVO_BASE = "servo_base"
    SERVO_LOWERARM = "servo_lowerarm"
    SERVO_FOREARM = "servo_forearm"
    SERVO_CLAW = "servo_claw"


class SerialCommunication:
    def __init__(self, port, baudrate=9600, timeout=1):
        self.serial_obj = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )
        time.sleep(2)

    def send_msg(self, servo_name:ServoName, angle:int):
        msg = f"{servo_name.value.strip()},{angle}\n"
        self.serial_obj.write(msg.encode())

        response = self.serial_obj.readline().decode().strip()
        print("Arduino:", response)

    def close(self):
        self.serial_obj.close()


def parse_xyz(s: str):
    """Parse 'x,y,z' (commas, spaces, or mix) into floats. Raises ValueError."""
    parts = [p for p in re.split(r'[,\s;]+', s.strip()) if p != '']
    if len(parts) != 3:
        raise ValueError(f"expected 3 values (x,y,z), got {len(parts)}: {s!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def main(): 
    serial_com = SerialCommunication(
        port="COM3",
        baudrate=9600,
        timeout=1
    )

    ik_solver = EezybotKinematics()

    print("Enter target X,Y,Z in cm, e.g. 23,0,21  (q to quit, c <angle> for claw)")
    try:
        while True:
            s = input("x,y,z cm > ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break

            # Claw manual override: "c 90" / "claw 90" / just "c" then prompt
            if s.lower().startswith("c"):
                rest = re.sub(r'^[Cc](law)?', '', s).strip(' ,:')
                if not rest:
                    rest = input("Claw angle (0-180): ").strip()
                try:
                    claw_angle = int(float(rest))
                except ValueError:
                    print(f"Bad claw angle: {rest!r}")
                    continue
                serial_com.send_msg(ServoName.SERVO_CLAW, claw_angle)
                continue

            try:
                x, y, z = parse_xyz(s)
            except ValueError as e:
                print(f"Bad input: {e}")
                continue

            # Pre-check reachability for a clean message (solve_ik also checks)
            ok, msg = ik_solver.is_reachable(x, y, z)
            if not ok:
                print(f"Unreachable ({x}, {y}, {z}): {msg}")
                continue

            try:
                theta1, theta2, theta3 = ik_solver.solve_ik(x, y, z)
            except ValueError as e:
                print(f"IK failed: {e}")
                continue

            m1, m2, m3 = ik_solver.joint_angles_to_servo_degrees(theta1, theta2, theta3)
            print(f"IK deg: base={math.degrees(theta1):.1f} shoulder={math.degrees(theta2):.1f} elbow_int={math.degrees(theta3):.1f}")
            print(f"Servo: base={m1} lowerarm={m2} forearm={m3}")

            serial_com.send_msg(ServoName.SERVO_BASE, m1)
            time.sleep(0.15)
            serial_com.send_msg(ServoName.SERVO_LOWERARM, m2)
            time.sleep(0.15)
            serial_com.send_msg(ServoName.SERVO_FOREARM, m3)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        serial_com.close()

if __name__ == "__main__":
    main()