#include <math.h>
#include "time.h"
#include "cv.h"
#include "highgui.h"

#define Total_train_face 32
#define Total_probe_face 32
#define Height 311
#define Width 232

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif