#include <stdio.h>
#include <math.h>

extern double calculate(double n, double no, double nhalf) {
    return exp(-(pow(((n-no)/(double)nhalf), 2.0)));
}

int main() {
    int n;
    int nhalf = 20;
    int no = nhalf*3;

    for (n = 0; n < 1000; n++) {
        double x = calculate(n, no, nhalf);
        if (isnan(x)) {
            printf("nan at %d\n", n);
        }
    }

    return 0;
}
