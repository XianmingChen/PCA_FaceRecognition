#include <stdio.h>
#include <stdlib.h>
#include "PCA.h"

void read_image(char *filepath,double image[][Width])
{
	int i,j;
	CvScalar s;
	IplImage *img=cvLoadImage(filepath,0);
    for(i=0;i<img->height;i++)
	{
        for(j=0;j<img->width;j++)
		{
			s=cvGet2D(img,i,j); 
			image[i][j]=s.val[0]/255;
        }
    }
    cvReleaseImage(&img);  
}

void load_image(char *filepath,CvMat *input,int rank)
{
	int i;
	CvMat row_header,*row,*mat;
	IplImage *img;

	img=cvLoadImage(filepath,0);
	mat=cvCreateMat(Height,Width,CV_32FC1);
	cvConvert(img,mat);
	row=cvReshape(mat,&row_header,0,1);

	float *ptr=(float*)(input->data.fl+rank*input->cols); //input->cols=Height*Width
	float *ptr2=(float*)row->data.fl;

	for (i=0;i<input->cols;i++)
	{
		*ptr=*ptr2;
		ptr++;
		ptr2++;
	}
	cvReleaseImage(&img);  
}

int PCA_Comparison(CvMat* training,CvMat* probing,int rank)
{
	int i,j;
	int index;
	float difference[Total_train_face]={0.0};
	float temp_difference;
	float* temp_train;
	float* temp_probe;

	for (i=0;i<training->rows;i++)
	{
		difference[i]=0.0;
		temp_train=(float*)training->data.fl+i*training->cols;
		temp_probe=(float*)probing->data.fl;
		for (j=0;j<probing->cols;j++)
		{
			difference[i]=(float)(*temp_train-*temp_probe)*(*temp_train-*temp_probe)+difference[i];
			temp_train++;
			temp_probe++;
		}
	}

	temp_difference=difference[0];
	index=0;
	for (i=1;i<training->rows;i++)
	{
		if (temp_difference>difference[i])
		{
			temp_difference=difference[i];
			index=i;
		}
	}

	if (index==rank)
		return 1;
	else 
		return 0;
}

int main()
{
	char image_path[255];
	int i;
	int Accuracy=0;
	int Probe_count=0;
	double total_start,total_end;
	FILE *record;

	record=fopen("result_PCA.txt","a+");
	total_start=clock();

	CvMat* training_space=cvCreateMat(Total_train_face,Height*Width,CV_32FC1);
	for (i=0;i<Total_train_face;i++)
	{
		printf("Loading image %d...\n",i+1);
		sprintf_s(image_path,"Aligned_FERET/input/trainfaces/%d.jpg",i+1);
		load_image(image_path,training_space,i);
	}

	printf("Begin to train images...\n");
	CvMat* average=cvCreateMat(1,training_space->cols,CV_32FC1);
	CvMat* eigen_values=cvCreateMat(1,min(training_space->rows,training_space->cols),CV_32FC1);
	CvMat* eigen_vectors=cvCreateMat(min(training_space->rows,training_space->cols),training_space->cols,CV_32FC1);
	cvCalcPCA(training_space,average,eigen_values,eigen_vectors,CV_PCA_DATA_AS_ROW);

	CvMat* training_result=cvCreateMat(training_space->rows,min(training_space->rows,training_space->cols),CV_32FC1);
	cvProjectPCA(training_space,average,eigen_vectors,training_result);

	CvMat* probing_space=cvCreateMat(1,Height*Width,CV_32FC1);
	CvMat* probing_result=cvCreateMat(probing_space->rows,min(training_space->rows,training_space->cols),CV_32FC1);

	printf("Begin to probe images...\n");
	for (i=0;i<Total_probe_face;i++)
	{
		printf("Image %d Probing...\n",i+1);
		sprintf_s(image_path,"Aligned_FERET/input/probefaces/%d.jpg",i+1);
		load_image(image_path,probing_space,0);
		cvProjectPCA(probing_space,average,eigen_vectors,probing_result);

		Accuracy=PCA_Comparison(training_result,probing_result,i);
		if (Accuracy==1)
		{
			Probe_count++;
			printf("Image %d Probe successfully!\n",i+1);
		}
		else
		{
			printf("Image %d Probe failed!\n",i+1);
		}
	}
	printf("%d Images probe completed!\n",Total_probe_face);
	printf("Probe accuracy= %f\n",(double)Probe_count/Total_probe_face);
	total_end=clock();
	total_start=total_end-total_start;
	printf("Running time for %d images is %f s\n",Total_probe_face,(float)(total_start/1000));
	fprintf(record,"Running time  for %d images is %f ms\n",Total_probe_face,total_start);
	fprintf(record,"The accuracy for %d images is %f\n",Total_probe_face,(double)Probe_count/Total_probe_face);
	fprintf(record,"\n");
	fclose(record);
	
	system("pause");
	return 0;
}