#include <Servo.h>

Servo baseServo;
Servo lowerServo;
Servo upperServo;
Servo clawServo;

const int BAUD_RATE = 9600;
const int CLAW_OPEN = 160;
const int CLAW_CLOSED = 100;

const int HOME_LOWER = 70;
const int HOME_BASE = 90;
const int HOME_UPPER = 70;

const int DROP_ODD_LOWER = 100;
const int DROP_ODD_BASE = 180;
const int DROP_ODD_UPPER = 100;

const int DROP_EVEN_LOWER = 100;
const int DROP_EVEN_BASE = 0;
const int DROP_EVEN_UPPER = 100;

struct Command {
  int id;
  int lower;
  int base;
  int upper;
};

void moveArm(int lowerAngle, int baseAngle, int upperAngle) {
  lowerServo.write(constrain(lowerAngle, 0, 180));
  baseServo.write(constrain(baseAngle, 0, 180));
  upperServo.write(constrain(upperAngle, 0, 180));
  delay(1000);
}

void setClaw(int angle) {
  clawServo.write(constrain(angle, 0, 180));
  delay(500);
}

void goHome() {
  moveArm(HOME_LOWER, HOME_BASE, HOME_UPPER);
}

bool parseCommand(String line, Command &cmd) {
  line.trim();
  if (line.length() == 0) {
    return false;
  }

  int firstComma = line.indexOf(',');
  if (firstComma < 0) {
    return false;
  }

  int secondComma = line.indexOf(',', firstComma + 1);
  if (secondComma < 0) {
    return false;
  }

  int thirdComma = line.indexOf(',', secondComma + 1);
  if (thirdComma < 0) {
    return false;
  }

  cmd.id = line.substring(0, firstComma).toInt();
  cmd.lower = line.substring(firstComma + 1, secondComma).toInt();
  cmd.base = line.substring(secondComma + 1, thirdComma).toInt();
  cmd.upper = line.substring(thirdComma + 1).toInt();
  // Basic sanity: marker id >= 0, servo angles 0..180.
  if (cmd.id < 0 || cmd.lower < 0 || cmd.lower > 180
      || cmd.base < 0 || cmd.base > 180
      || cmd.upper < 0 || cmd.upper > 180) {
    return false;
  }
  return true;
}

void doPickAndDrop(Command cmd) {
  // 1. Grab: claw guaranteed open, move to mapped pick pose, close.
  setClaw(CLAW_OPEN);
  moveArm(cmd.lower, cmd.base, cmd.upper);
  setClaw(CLAW_CLOSED);
  delay(350);

  // 2. Lift/retreat to home position (lower=70, base=90, upper=70).
  goHome();

  // 3. Sort by marker id parity.
  if (cmd.id % 2 == 1) {
    // Odd -> drop zone 1: lower=100, base=180, upper=100.
    moveArm(DROP_ODD_LOWER, DROP_ODD_BASE, DROP_ODD_UPPER);
  } else {
    // Even -> drop zone 2: lower=100, base=0, upper=100.
    moveArm(DROP_EVEN_LOWER, DROP_EVEN_BASE, DROP_EVEN_UPPER);
  }

  // 4. Release and return to home position with claw open.
  setClaw(CLAW_OPEN);
  delay(350);
  goHome();
}

void setup() {
  Serial.begin(BAUD_RATE);

  // Hold initial pose from the very first pulse.
  // Servo.h defaults to 90 deg on attach(), which jerks the arm away
  // from the physical initial pose (70, 90, 70) before goHome() pulls
  // it back. write() BEFORE attach() preloads the correct pulse width
  // so the first pulse out already is the home pose -> no initial jump.
  lowerServo.write(HOME_LOWER);
  baseServo.write(HOME_BASE);
  upperServo.write(HOME_UPPER);
  clawServo.write(CLAW_OPEN);

  baseServo.attach(10);
  lowerServo.attach(11);
  upperServo.attach(9);
  clawServo.attach(6);

  setClaw(CLAW_OPEN);
  goHome();
}

void loop() {
  if (Serial.available() > 0) {
    String packet = Serial.readStringUntil('\n');
    packet.trim();

    if (packet.length() > 0) {
      Command cmd;
      if (parseCommand(packet, cmd)) {
        doPickAndDrop(cmd);
      }
    }

    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}

// Run the servo angle mapper
//python3 servo_angle_mapper.py --port=/dev/ttyACM0 --baud=9600