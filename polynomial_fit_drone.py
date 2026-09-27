import numpy as np
import matplotlib.pyplot as plt

def build_vandermonde(x, degree):
    n = len(x)
    A = np.zeros((n, degree + 1))
    for i in range(n):
        for j in range(degree + 1):
            A[i][j] = x[i] ** j
    return A


def normal_equation_matrices(A, y):
    At = A.T
    AtA = At @ A
    Aty = At @ y
    return AtA, Aty


def gaussian_elimination(M, b):
    k = len(b)
    Aug = np.hstack([M.astype(float), b.reshape(-1, 1).astype(float)])
    for col in range(k):
        pivot_row = np.argmax(np.abs(Aug[col:, col])) + col
        if Aug[pivot_row, col] == 0:
            raise ValueError("เมทริกซ์เอกฐาน (singular matrix) ไม่สามารถแก้ได้")
        Aug[[col, pivot_row]] = Aug[[pivot_row, col]]
        for row in range(col + 1, k):
            factor = Aug[row, col] / Aug[col, col]
            Aug[row, col:] -= factor * Aug[col, col:]
    c = np.zeros(k)
    for row in range(k - 1, -1, -1):
        c[row] = (Aug[row, -1] - Aug[row, row + 1:k] @ c[row + 1:k]) / Aug[row, row]
    return c


def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def format_polynomial(coeffs, decimals=3):
    terms = []
    for j, a in enumerate(coeffs):
        if j == 0:
            terms.append(f"{a:.{decimals}f}")
        elif j == 1:
            terms.append(f"{a:+.{decimals}f}x")
        else:
            terms.append(f"{a:+.{decimals}f}x^{j}")
    return "y = " + " ".join(terms)


def predict(coeffs, x_new):
    return sum(a * (x_new ** j) for j, a in enumerate(coeffs))

if __name__ == "__main__":
    # X = เวลาหลังปล่อยโดรน (วินาที)
    # Y = ความสูงที่วัดได้จากเซนเซอร์วัดความดัน (เมตร)
    x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    y = np.array([0.00, 1.05, 1.95, 3.30, 4.65, 6.40, 8.15,10.30, 12.40, 15.05, 17.55])

    degree = 2 
    print("ข้อมูลนำเข้า (การปล่อยโดรนขึ้นบิน)")
    for xi, yi in zip(x, y):
        print(f"  t = {xi:>4} s   h = {yi:>6} m")

    A = build_vandermonde(x, degree)
    AtA, Aty = normal_equation_matrices(A, y)
    coeffs = gaussian_elimination(AtA, Aty)
    y_pred = A @ coeffs
    r2 = r_squared(y, y_pred)

    print("\nผลลัพธ์สมการพหุนาม")
    print(format_polynomial(coeffs))
    # print(f"ค่าความแม่นยำของแบบจำลอง (R^2) = {r2:.6f}")
    t_test = 6
    h_test = predict(coeffs, t_test)
    print(f"\nพยากรณ์ความสูงที่ t = {t_test} วินาที -> h = {round(h_test, 3)} เมตร")

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="crimson", label="Measured data (sensor)", zorder=3)
    xs = np.linspace(0, 6, 200)
    ys = predict(coeffs, xs)
    plt.plot(xs, ys, color="navy", label="Fitted polynomial (degree 2)")
    plt.axvline(5, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Time t (seconds)")
    plt.ylabel("Altitude h (meters)")
    plt.title("Drone Launch: Time vs Altitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("drone.png", dpi=150)
