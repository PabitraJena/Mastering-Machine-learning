all:
	gcc 6.c -o a.exe -Wall -Wextra
opt:
	gcc -O3 -march=native -ffast-math -DNDEBUG 6.c -o a.exe
