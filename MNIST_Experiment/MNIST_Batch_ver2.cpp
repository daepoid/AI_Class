//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include <string.h>
//#include <ctype.h>
//#include <math.h>
//#pragma warning(disable:4996)
//
//#define Sz0 784
//#define Sz1	500
//#define Sz2	400
//#define Sz3	10
//#define Ori_learning_rate 0.05
//
//#define MAX_epoch	20
//#define Num_train	60000
//#define Num_test 10000
//#define Size_minibatch 4
//
//double F0[Sz0];
//double F1[Sz1];
//double F2[Sz2];
//double F3[Sz3];
//
//double IN1[Sz0];
//double IN2[Sz1];
//double IN3[Sz2];
//
//double W1_gradient[Sz0][Sz1];
//double W2_gradient[Sz1][Sz2];
//double W3_gradient[Sz2][Sz3];
//double B1_gradient[Sz1];
//double B2_gradient[Sz2];
//double B3_gradient[Sz3];
//
//double W1[Sz0][Sz1];
//double W2[Sz1][Sz2];
//double W3[Sz2][Sz3];
//double B1[Sz1];
//double B2[Sz2];
//double B3[Sz3];
//double SoftMax[Sz3];
//
//double Delta1[Sz1];
//double Delta2[Sz2];
//double Delta3[Sz3];
//
//double sigmoid_deriv1[Sz1];
//double sigmoid_deriv2[Sz2];
//
//double Train_Data[Num_train][28][28];
//int Trian_Label_ori[Num_train];
//
//double learning_rate = Ori_learning_rate;
//
//double Test_Data[Num_test][28][28];
//int Test_Label[Num_test];
//
//FILE* fp_accuracy;
//
//void Read_training_data();
//void Read_test_data();
//void Init_Parameter();
//double getRandNum();
//void Do_train();
//double Do_test();
//double Compute_sigmoid_F1_from_F0(int i);
//double Compute_sigmoid_F2_from_F1(int i);
//double Compute_IN3_from_F2(int i);
//
//int main() {
//	fp_accuracy = fopen("Test_Accuracy.txt", "a");
//	if (!fp_accuracy) {
//		printf("FILE opne failed : Text_Accuracy.txt\n");
//		return 0;
//	}
//
//	srand(time(NULL));
//
//	fprintf(fp_accuracy, "\n\nExperiment of building Supervised NN system has started.\n");
//	fprintf(fp_accuracy, "Sz0 : %5d, Sz1 : %5d, Sz2 : %5d, Sz3 : %5d, Num_Train_exampls : %6d, Num_Test_exampls : %6d, Ori_learning_rate : %f, Size_minibatch : %d\n", Sz0, Sz1, Sz2, Sz3, Num_train, Num_test, Ori_learning_rate, Size_minibatch);
//	fprintf(fp_accuracy, "/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n\n");
//	fflush(fp_accuracy);
//
//	Read_training_data();
//	Read_test_data();
//
//	Init_Parameter();
//
//	Do_train();
//
//	fclose(fp_accuracy);
//
//	return 1;
//}
//
//void Read_training_data() {
//	FILE* fp;
//	char* ret_fgets;
//	char buff_line[1024];
//	char num_str[20];
//	int leng;
//
//	fp = fopen("datasets/train.txt", "r");
//	if (!fp) {
//		printf("File open failed : mnist_train_data.txt\n");
//		return;
//	}
//
//	printf("Training data loading start!\n");
//
//	for (int testCase = 0; testCase < Num_train; testCase++) {
//		ret_fgets = fgets(buff_line, 1024, fp);
//		if (!ret_fgets) {
//			printf("fgets error\n");
//			return;
//		}
//		leng = strlen(buff_line);
//		if (leng != 2) {
//			printf("Wrong length of label\n");
//			getchar();
//		}
//		buff_line[--leng] = '\0';
//		Trian_Label_ori[testCase] = atoi(buff_line);
//
//		for (int r = 0; r < 28; r++) {
//			ret_fgets = fgets(buff_line, 1024, fp);
//			if (!ret_fgets) {
//				printf("fgets error\n");
//				return;
//			}
//
//			leng = strlen(buff_line);
//			buff_line[--leng] = '\0';
//
//			int cp = 0;
//			for (int c = 0; c < 28; c++) {
//				while (buff_line[cp] == ' ')
//					cp++;
//				int i = 0;
//				while (isdigit(buff_line[cp])) {
//					num_str[i] = buff_line[cp];
//					i++;
//					cp++;
//				}
//				num_str[i] = '\0';
//				Train_Data[testCase][r][c] = atof(num_str);
//			}// end c for loop
//		}// end r for loop
//	}// end tastCase for loop
//	printf("Training data loading finish.\n");
//}
//
//void Read_test_data() {
//	FILE* fp;
//	char* ret_fgets;
//	char buff_line[1024];
//	char num_str[20];
//	int leng;
//
//	fp = fopen("datasets/test.txt", "r");
//	if (!fp) {
//		printf("File open failed : mnist_train_data.txt\n");
//		return;
//	}
//
//	printf("Training data loading start!\n");
//
//	for (int testCase = 0; testCase < Num_test; testCase++) {
//		ret_fgets = fgets(buff_line, 1024, fp);
//		if (!ret_fgets) {
//			printf("fgets error\n");
//			return;
//		}
//		leng = strlen(buff_line);
//		if (leng != 2) {
//			printf("Wrong length of label\n");
//			getchar();
//		}
//		buff_line[--leng] = '\0';
//		Test_Label[testCase] = atoi(buff_line);
//
//		for (int r = 0; r < 28; r++) {
//			ret_fgets = fgets(buff_line, 1024, fp);
//			if (!ret_fgets) {
//				printf("fgets error\n");
//				return;
//			}
//
//			leng = strlen(buff_line);
//			buff_line[--leng] = '\0';
//
//			int cp = 0;
//			for (int c = 0; c < 28; c++) {
//				while (buff_line[cp] == ' ')
//					cp++;
//				int i = 0;
//				while (isdigit(buff_line[cp])) {
//					num_str[i] = buff_line[cp];
//					i++;
//					cp++;
//				}
//				num_str[i] = '\0';
//				Test_Data[testCase][r][c] = atof(num_str);
//			}// end c for loop
//		}// end r for loop
//	}// end tastCase for loop
//	printf("Test data loading finish.\n");
//}
//
//void Init_Parameter() {
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1[i][j] = getRandNum();
//		}
//	}
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2[i][j] = getRandNum();
//		}
//	}
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3[i][j] = getRandNum();
//		}
//	}
//
//	for (int i = 0; i < Sz3; i++) {
//		B3[i] = getRandNum();
//	}
//	for (int i = 0; i < Sz2; i++) {
//		B2[i] = getRandNum();
//	}
//	for (int i = 0; i < Sz1; i++) {
//		B1[i] = getRandNum();
//	}
//}
//
//void initialize_gradients_before_minibatch() {
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1_gradient[i][j] = 0;
//		}
//	}
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2_gradient[i][j] = 0;
//		}
//	}
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3_gradient[i][j] = 0;
//		}
//	}
//
//	for (int i = 0; i < Sz3; i++) {
//		B3_gradient[i] = 0;
//	}
//	for (int i = 0; i < Sz2; i++) {
//		B2_gradient[i] = 0;
//	}
//	for (int i = 0; i < Sz1; i++) {
//		B1_gradient[i] = 0;
//	}
//}
//
//double getRandNum() {
//	double r, sign;
//	int r2;
//
//	r = (double)rand();
//	r2 = rand();
//	if (r2 < RAND_MAX / 2)
//		sign = -1.0;
//	else
//		sign = 1.0;
//
//	return (r / (double)RAND_MAX) * sign;
//}
//
//void forward_compute(int testCase) {
//	int j = 0;
//	for (int r = 0; r < 28; r++) {
//		for (int c = 0; c < 28; c++) {
//			F0[j] = Train_Data[testCase][r][c] / 255;
//			j++;
//		}// end c for loop
//	}// end r for loop
//
//	for (int i = 0; i < Sz1; i++) {
//		F1[i] = Compute_sigmoid_F1_from_F0(i);
//		sigmoid_deriv1[i] = F1[i] * (1 - F1[i]);
//	}
//
//	for (int i = 0; i < Sz2; i++) {
//		F2[i] = Compute_sigmoid_F2_from_F1(i);
//		sigmoid_deriv2[i] = F2[i] * (1 - F2[i]);
//	}
//
//	for (int i = 0; i < Sz3; i++) {
//		F3[i] = Compute_IN3_from_F2(i);
//	}
//
//	double denomi = 0.0;
//	for (int i = 0; i < Sz3; i++) {
//		denomi = denomi + exp(F3[i]);
//	}
//
//	for (int i = 0; i < Sz3; i++) {
//		SoftMax[i]= exp(F3[i]) / denomi;
//	}
//}
//
//void backward_compute(int testCase) {
//	int correct_label;
//
//	for (int i = 0; i < Sz3; i++) {
//		Delta3[i] = SoftMax[i];
//	}
//
//	// if i=j 
//	correct_label = Trian_Label_ori[testCase];
//	Delta3[correct_label] = Delta3[correct_label] - 1;
//
//	for (int j = 0; j < Sz2; j++) {
//		double temp = 0;
//		for (int k = 0; k < Sz3; k++) {
//			temp = temp + Delta3[k] * W3[j][k];
//		}
//		Delta2[j] = sigmoid_deriv2[j] * temp;
//	}
//
//	for (int j = 0; j < Sz1; j++) {
//		double temp = 0;
//		for (int k = 0; k < Sz2; k++) {
//			temp = temp + Delta2[k] * W2[j][k];
//		}
//		Delta1[j] = sigmoid_deriv1[j] * temp;
//	}
//}
//
//void update_parameters_for_minibatch() {
//
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3[i][j] = W3[i][j] - learning_rate * W3_gradient[i][j];
//		}
//	}
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2[i][j] = W2[i][j] - learning_rate * W2_gradient[i][j];
//		}
//	}
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1[i][j] = W1[i][j] - learning_rate * W1_gradient[i][j];;
//		}
//	}
//
//	for (int i = 0; i < Sz3; i++) {
//		B3[i] = B3[i] - learning_rate * B3_gradient[i];
//	}
//	for (int i = 0; i < Sz2; i++) {
//		B2[i] = B2[i] - learning_rate * B2_gradient[i];;
//	}
//	for (int i = 0; i < Sz1; i++) {
//		B1[i] = B1[i] - learning_rate * B1_gradient[i];;
//	}
//}
//
//void Accumulate_gradients_of_parameters(int testCase) {
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3_gradient[i][j] += Delta3[j] * F2[i];
//		}
//	}
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2_gradient[i][j] += Delta2[j] * F1[i];
//		}
//	}
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1_gradient[i][j] += Delta1[j] * F0[i];
//		}
//	}
//
//	for (int i = 0; i < Sz3; i++) {
//		B3_gradient[i] += Delta3[i];
//	}
//	for (int i = 0; i < Sz2; i++) {
//		B2_gradient[i] += Delta2[i];
//	}
//	for (int i = 0; i < Sz1; i++) {
//		B1_gradient[i] += Delta1[i];
//	}
//}
//
//void Do_train() {
//	printf("\n\n///////////////Start NN Training use minibatch!///////////////\n");
//
//	int idx_miniBatch, start_of_this_minibatch, start_of_next_minibatch,count =0;
//	for (int epoch = 0; epoch < MAX_epoch; epoch++) {
//		idx_miniBatch = 0;
//		do {
//			if (idx_miniBatch % 500 == 0)
//				printf("Running NN // idx_minibatch == %d\n", idx_miniBatch);
//
//			start_of_this_minibatch = idx_miniBatch * Size_minibatch;
//			start_of_next_minibatch = start_of_this_minibatch + Size_minibatch;
//			if (!(start_of_next_minibatch - 1 <= Num_train - 1)) {
//				break;
//			}
//
//			initialize_gradients_before_minibatch();
//			for (int testCase = start_of_this_minibatch; testCase < start_of_next_minibatch; testCase++) {
//				forward_compute(testCase);
//				backward_compute(testCase);
//				Accumulate_gradients_of_parameters(testCase);
//			}
//			update_parameters_for_minibatch();
//			idx_miniBatch += 1;
//		} while (1);
//
//		printf("\nSupervised NN system Testing after epoch = %d has started.\n", epoch);
//		double Accuracy = Do_test();
//		fprintf(fp_accuracy, "After epoch : %d, learning_rate : %f, S-NN Accuracy = %f\n", epoch, learning_rate, Accuracy);
//		fflush(fp_accuracy);
//		printf("S-NN Accuracy After epoch %3d = %f (stored in file Test_Accuracy.txt)\n\n", epoch, Accuracy);
//	}
//}
//
//double Do_test() {
//	double temp;
//	int correct_cnt = 0;
//	for (int testCase = 0; testCase < Num_test; testCase++) {
//		if (testCase % 1000 == 0)
//			printf("Test example Number : %d\n", testCase);
//		for (int r = 0; r < 28; r++) {
//			for (int c = 0; c < 28; c++) {
//				F0[r * 28 + c] = Test_Data[testCase][r][c] / 255;
//			}// end c for loop
//		}// end r for loop
//
//		for (int i = 0; i < Sz1; i++) {
//			F1[i] = Compute_sigmoid_F1_from_F0(i);
//		}
//		for (int i = 0; i < Sz2; i++) {
//			F2[i] = Compute_sigmoid_F2_from_F1(i);
//		}
//		for (int i = 0; i < Sz3; i++) {
//			F3[i] = Compute_IN3_from_F2(i);
//		}
//
//		temp = 0.0;
//		for (int i = 0; i < Sz3; i++) {
//			temp = temp + exp(F3[i]);
//		}
//
//		for (int i = 0; i < Sz3; i++) {
//			SoftMax[i] = exp(F3[i]) / temp;
//		}
//
//		int max_idx = 0;
//		double max_val = SoftMax[0];
//		for (int i = 1; i < Sz3; i++) {
//			if (max_val < SoftMax[i]) {
//				max_idx = i;
//				max_val = SoftMax[i];
//			}
//		}
//
//		int Label_by_model = max_idx;
//    
//		if (Label_by_model == Test_Label[testCase])
//			correct_cnt++;
//	}
//	double accuracy = ((double)correct_cnt) / Num_test;
//	return accuracy;
//}
//
//double Compute_sigmoid_F1_from_F0(int i) {
//	double res, x = 0;
//
//	for (int j = 0; j < Sz0; j++)
//		x += W1[j][i] * F0[j];
//	x += B1[i];
//	IN1[i] = x;
//	res = 1.0 / (1 + exp(-x));
//	return res;
//}
//
//double Compute_sigmoid_F2_from_F1(int i) {
//	double res, x = 0;
//
//	for (int j = 0; j < Sz1; j++)
//		x += W2[j][i] * F1[j];
//	x += B2[i];
//	IN2[i] = x;
//	res = 1.0 / (1 + exp(-x));
//	return res;
//}
//
//double Compute_IN3_from_F2(int i) {
//	double res, x = 0;
//
//	for (int j = 0; j < Sz2; j++)
//		x += W3[j][i] * F2[j];
//	x += B3[i];
//	IN1[i] = x;
//	return x;
//}