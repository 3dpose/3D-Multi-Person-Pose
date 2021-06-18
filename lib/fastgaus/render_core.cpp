#include "render_core.h"

void _get_gaus(float* map, int sizeh, int sizew, float sigma, float x, float y){
	int i,j,c;
	float thresh = - log(0.01);
	int idx = 0;
	float e,dx,dy;
	for(i=0;i<sizeh;i++){
		for (j=0;j<sizew;j++){
			dx = float(j) - x;
			dy = float(i) - y;
			e = (dx*dx + dy*dy) / 2 / sigma/ sigma;
			if(e<thresh){
				map[idx] = exp(-e);
			}
			idx++;
		}
	}
}