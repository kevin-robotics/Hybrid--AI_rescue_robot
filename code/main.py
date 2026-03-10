import time
import math
import cv2
import RPi.GPIO as GPIO
import numpy as np

from motor import Motor
from ultrasonic import Ultrasonic
from alert import AlertSystem
from camera_ai import CameraAI
from fuzzy_controller import FuzzyController
from pso_nmpc import PSONMPC

STOP_DISTANCE = 20  # cm

def main():
    GPIO.setmode(GPIO.BCM)

    motor       = Motor(22, 23, 17, 18)
    ultrasonic  = Ultrasonic(24, 25)
    alert       = AlertSystem(5, 6)
    camera      = CameraAI("model/detect.tflite")
    fuzzy       = FuzzyController(STOP_DISTANCE, 70)

    #  PSO–NMPC controller
    nmpc = PSONMPC(
        dt=0.1,
        horizon=10,
        num_particles=30,
        num_iterations=20,
        w_inertia=0.5,
        c1=1.5,
        c2=1.5,
        #  ref_path=np.vstack((..., ...)).T
    )

    # state estimate & counter
    x = y = theta = 0.0
    step_index = 0

    try:
        print(" Robot running. Press Ctrl+C to stop.")
        while True:
            frame = camera.get_frame()
            detected, boxes = camera.detect_person(frame)

            if detected:
                camera.draw_boxes(frame, boxes)
                print(" Person detected → STOP & ALERT")
                motor.stop()
                alert.alert_on()

            else:
                alert.alert_off()
                dist = ultrasonic.get_distance() or 100
                dist = min(dist, 100)

                if dist < STOP_DISTANCE:
                    # --- AVOID MANEUVER ---
                    print(f" Obstacle at {dist:.1f} cm → Avoiding")
                    motor.move(0, 70)    # turn right
                    time.sleep(0.5)
                    motor.move(70, 70)   # forward
                    time.sleep(0.7)
                    motor.move(70, 0)    # turn left
                    time.sleep(0.5)
                    
                    continue

                # --- RETURN TO TRAJECTORY ---
                
                dists = np.linalg.norm(nmpc.ref_path - np.array([x, y]), axis=1)
                si = int(np.argmin(dists))
                
                step_index = min(si, len(nmpc.ref_path) - nmpc.N)

                # --- RUN PSO-NMPC ---
                try:
                    v_opt, w_opt = nmpc.optimize(x, y, theta, step_index)
                except Exception as e:
                    print(" NMPC error, falling back to fuzzy:")
                    ls, rs = fuzzy.compute_speed(dist)
                    motor.move(ls, rs)
                    continue

                # normalize and map to PWM
                V_MAX, W_MAX = 1.0, 2.0
                vn = max(-1, min(1, v_opt / V_MAX))
                wn = max(-1, min(1, w_opt / W_MAX))

                BASE, FWD_SCALE, TURN_SCALE = 50, 40, 30
                left_pwm  = BASE + FWD_SCALE * vn - TURN_SCALE * wn
                right_pwm = BASE + FWD_SCALE * vn + TURN_SCALE * wn
                left_pwm  = max(0, min(100, left_pwm))
                right_pwm = max(0, min(100, right_pwm))

                print(f"NMPC → v={v_opt:.2f}, w={w_opt:.2f}  PWM L={left_pwm:.0f}, R={right_pwm:.0f}")
                motor.move(left_pwm, right_pwm)

                # update internal state and index
                x     += v_opt * math.cos(theta) * nmpc.dt
                y     += v_opt * math.sin(theta) * nmpc.dt
                theta = (theta + w_opt * nmpc.dt + math.pi) % (2*math.pi) - math.pi
                step_index += 1

            # display camera & boxes
            camera.draw_boxes(frame, boxes)
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print(" Program interrupted. Cleaning up...")

    finally:
        motor.stop()
        alert.cleanup()
        camera.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
