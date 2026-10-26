import numpy as np

def main():
    np.random.seed(42); matrix = np.random.randint(10, 99, size=(4,4))
    print('Original matrix:\n', matrix)
    print('Row 2:', matrix[1,:]); print('Column 3:', matrix[:,2]); print('Top-right block:\n', matrix[:2,2:])
    print('Global mean:', np.mean(matrix)); print('Column sums:', np.sum(matrix, axis=0)); print('Row maxima:', np.max(matrix, axis=1))
    print('Squared:\n', np.square(matrix)); print('Transpose:\n', matrix.T); print('A @ A.T:\n', matrix @ matrix.T); print('Determinant:', np.linalg.det(matrix.astype(float)))

if __name__ == '__main__': main()
