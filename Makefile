all:
	gcc 7.c -o a.exe -Wall -Wextra
opt:
	gcc -O3 -march=native -ffast-math -DNDEBUG 7.c -o a.exe
