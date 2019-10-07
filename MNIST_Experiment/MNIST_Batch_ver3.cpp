//#include <iostream>
//#include <cstdlib>
//#include <ctime>
//#include <cstring>
//#include <cctype>
//#include <cmath>
//#pragma warning(disable:4996)
//
//#define SZ0 784
//#define SZ1	500
//#define SZ2	400
//#define SZ3	10
//#define LEARNING_RATE 0.05
//
//#define MAX_EPOCH	5
//#define TRAIN_SIZE	60000
//#define TEST_SIZE 10000
//#define SIZE_MINIBATCH 8
//
//#define ROW 28
//#define COLUMN 28
//
//double F0[SZ0];
//double F1[SZ1];
//double F2[SZ2];
//double F3[SZ3];
//
//double IN1[SZ0];
//double IN2[SZ1];
//double IN3[SZ2];
//
//double W1_gradient[SZ0][SZ1];
//double W2_gradient[SZ1][SZ2];
//double W3_gradient[SZ2][SZ3];
//double B1_gradient[SZ1];
//double B2_gradient[SZ2];
//double B3_gradient[SZ3];
//
//double W1[SZ0][SZ1];
//double W2[SZ1][SZ2];
//double W3[SZ2][SZ3];
//double B1[SZ1];
//double B2[SZ2];
//double B3[SZ3];
//double SoftMax[SZ3];
//
//double Delta1[SZ1];
//double Delta2[SZ2];
//double Delta3[SZ3];
//
//double sigmoid_deriv1[SZ1];
//double sigmoid_deriv2[SZ2];
//
//double Train_Data[TRAIN_SIZE][ROW][COLUMN];
//int Train_Label[TRAIN_SIZE];
//
//double learning_rate = LEARNING_RATE;
//
//double Test_Data[TEST_SIZE][ROW][COLUMN];
//int Test_Label[TEST_SIZE];
//
//int minibatch_size = SIZE_MINIBATCH;
//
//FILE* fp_accuracy;
//
//void Read_training_data();
//void Read_test_data();
//void init_parameters();
//void Do_train();
//double Do_test();
//double getRandNum();
//double Compute_sigmoid_F1_from_F0(int i);
//double Compute_sigmoid_F2_from_F1(int i);
//double Compute_IN3_from_F2(int i);
//
//int main() {
//	fp_accuracy = fopen("Test_Accuracy_Batch_ver3.txt", "a");
//	if (!fp_accuracy) {
//		printf("FILE opne failed : Text_Accuracy.txt\n");
//		return 0;
//	}
//
//	srand(time(NULL));
//
//  fprintf(fp_accuracy, "\n\nExperiment of building Supervised NN system has started.\n");
//  fprintf(fp_accuracy, "sz0 = %5d, sz1 = %5d, sz2 = %5d, sz3 = %5d, train_set_size = %6d, test_set_size = %6d, learning_rate = %lf\n", 
//    SZ0, SZ1, SZ2, SZ3, TRAIN_SIZE, TEST_SIZE, learning_rate);
//  fflush(fp_accuracy);
//  printf("Test_Accuracy_Batch_ver3 Start\n");
//  printf("Read train data\n");
//	Read_training_data();
//  printf("Read test data\n"); 
//	Read_test_data();
//
//  ////////////////////////////////////////////////////
//  printf("Init Parameter\n");
//	init_parameters();
//  printf("Do Train\n");
//	Do_train();
//
//  ////////////////////////////////////////////////////
//  minibatch_size = minibatch_size / 2; // 4
//  printf("Init Parameter\n");
//  init_parameters();
//  printf("Do Train\n");
//  Do_train();
//
//  ////////////////////////////////////////////////////
//  minibatch_size = minibatch_size / 2; // 2
//  printf("Init Parameter\n");
//  init_parameters();
//  printf("Do Train\n");
//  Do_train();
//
//
//  ////////////////////////////////////////////////////
//  minibatch_size = SIZE_MINIBATCH;
//  learning_rate = 0.05;
//  printf("Init Parameter\n");
//  init_parameters();
//  printf("Do Train\n");
//  Do_train();
//
//  ////////////////////////////////////////////////////
//  learning_rate = 0.1;
//  printf("Init Parameter\n");
//  init_parameters();
//  printf("Do Train\n");
//  Do_train();
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
//		printf("File open failed : train.txt\n");
//		return;
//	}
//
//	for (int index = 0; index < TRAIN_SIZE; index++) {
//    if (index % 10000 == 0) {
//      printf("Train data loading. train_index = %d\n", index);
//    }
//		ret_fgets = fgets(buff_line, 1024, fp);
//		if (!ret_fgets) {
//			printf("fgets error1.\n");
//			return;
//		}
//		leng = strlen(buff_line);
//		if (leng != 2) {
//      printf("wrong leng of error.\n");
//			getchar();
//		}
//		buff_line[--leng] = '\0';
//		Train_Label[index] = atoi(buff_line);
//
//		for (int row = 0; row < ROW; row++) {
//			ret_fgets = fgets(buff_line, 1024, fp);
//			if (!ret_fgets) {
//				printf("fgets error2.\n");
//				return;
//			}
//
//			leng = strlen(buff_line);
//			buff_line[--leng] = '\0';
//
//			int cp = 0;
//			for (int column = 0; column < COLUMN; column++) {
//				while (buff_line[cp] == ' ')
//					cp++;
//				int i = 0;
//				while (isdigit(buff_line[cp])) {
//					num_str[i] = buff_line[cp];
//					i++;
//					cp++;
//				}
//				num_str[i] = '\0';
//				Train_Data[index][row][column] = atof(num_str);
//			}
//		}
//	}
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
//	for (int index = 0; index < TEST_SIZE; index++) {
//    if (index % 2000 == 0) {
//      printf("Test data loading. test_index = %d\n", index);
//    }
//		ret_fgets = fgets(buff_line, 1024, fp);
//		if (!ret_fgets) {
//			printf("fgets error3.\n");
//			return;
//		}
//		leng = strlen(buff_line);
//		if (leng != 2) {
//			printf("Wrong length of label\n");
//			getchar();
//		}
//		buff_line[--leng] = '\0';
//		Test_Label[index] = atoi(buff_line);
//
//		for (int row = 0; row < ROW; row++) {
//			ret_fgets = fgets(buff_line, 1024, fp);
//			if (!ret_fgets) {
//				printf("fgets error4.\n");
//				return;
//			}
//
//			leng = strlen(buff_line);
//			buff_line[--leng] = '\0';
//
//			int cp = 0;
//			for (int column = 0; column < COLUMN; column++) {
//				while (buff_line[cp] == ' ')
//					cp++;
//				int i = 0;
//				while (isdigit(buff_line[cp])) {
//					num_str[i] = buff_line[cp];
//					i++;
//					cp++;
//				}
//				num_str[i] = '\0';
//				Test_Data[index][row][column] = atof(num_str);
//			}
//		}
//	}
//}
//
//void init_parameters() {
//	for (int i = 0; i < SZ0; i++) {
//		for (int j = 0; j < SZ1; j++) {
//			W1[i][j] = getRandNum();
//		}
//	}
//	for (int i = 0; i < SZ1; i++) {
//		for (int j = 0; j < SZ2; j++) {
//			W2[i][j] = getRandNum();
//		}
//	}
//	for (int i = 0; i < SZ2; i++) {
//		for (int j = 0; j < SZ3; j++) {
//			W3[i][j] = getRandNum();
//		}
//	}
//
//	for (int i = 0; i < SZ3; i++) {
//		B3[i] = getRandNum();
//	}
//	for (int i = 0; i < SZ2; i++) {
//		B2[i] = getRandNum();
//	}
//	for (int i = 0; i < SZ1; i++) {
//		B1[i] = getRandNum();
//	}
//}
//
//void initialize_gradients_before_minibatch() {
//	for (int i = 0; i < SZ0; i++) {
//		for (int j = 0; j < SZ1; j++) {
//			W1_gradient[i][j] = 0;
//		}
//	}
//	for (int i = 0; i < SZ1; i++) {
//		for (int j = 0; j < SZ2; j++) {
//			W2_gradient[i][j] = 0;
//		}
//	}
//	for (int i = 0; i < SZ2; i++) {
//		for (int j = 0; j < SZ3; j++) {
//			W3_gradient[i][j] = 0;
//		}
//	}
//
//	for (int i = 0; i < SZ3; i++) {
//		B3_gradient[i] = 0;
//	}
//	for (int i = 0; i < SZ2; i++) {
//		B2_gradient[i] = 0;
//	}
//	for (int i = 0; i < SZ1; i++) {
//		B1_gradient[i] = 0;
//	}
//}
//
//double getRandNum() {
//	double r, sign = 1.0;
//	int r2;
//
//	r = (double)rand();
//	r2 = rand();
//  if (r2 < RAND_MAX / 2) {
//    sign = -1.0;
//  }
//	return (r / (double)RAND_MAX) * sign;
//}
//
//void forward_compute(int index) {
//	int j = 0;
//	for (int row = 0; row < 28; row++) {
//		for (int column = 0; column < 28; column++) {
//			F0[j] = Train_Data[index][row][column] / 255;
//			j++;
//		}
//	}
//
//	for (int i = 0; i < SZ1; i++) {
//		F1[i] = Compute_sigmoid_F1_from_F0(i);
//		sigmoid_deriv1[i] = F1[i] * (1 - F1[i]);
//	}
//
//	for (int i = 0; i < SZ2; i++) {
//		F2[i] = Compute_sigmoid_F2_from_F1(i);
//		sigmoid_deriv2[i] = F2[i] * (1 - F2[i]);
//	}
//
//	for (int i = 0; i < SZ3; i++) {
//		F3[i] = Compute_IN3_from_F2(i);
//	}
//
//	double denomi = 0.0;
//	for (int i = 0; i < SZ3; i++) {
//		denomi = denomi + exp(F3[i]);
//	}
//
//	for (int i = 0; i < SZ3; i++) {
//		SoftMax[i]= exp(F3[i]) / denomi;
//	}
//}
//
//void backward_compute(int index) {
//	int correct_label;
//
//	for (int i = 0; i < SZ3; i++) {
//		Delta3[i] = SoftMax[i];
//	}
//
//	correct_label = Train_Label[index];
//	Delta3[correct_label] = Delta3[correct_label] - 1;
//
//	for (int j = 0; j < SZ2; j++) {
//		double temp = 0;
//		for (int k = 0; k < SZ3; k++) {
//			temp = temp + Delta3[k] * W3[j][k];
//		}
//		Delta2[j] = sigmoid_deriv2[j] * temp;
//	}
//
//	for (int j = 0; j < SZ1; j++) {
//		double temp = 0;
//		for (int k = 0; k < SZ2; k++) {
//			temp = temp + Delta2[k] * W2[j][k];
//		}
//		Delta1[j] = sigmoid_deriv1[j] * temp;
//	}
//}
//
//void update_parameters_for_minibatch() {
//
//	for (int i = 0; i < SZ2; i++) {
//		for (int j = 0; j < SZ3; j++) {
//			W3[i][j] = W3[i][j] - learning_rate * W3_gradient[i][j];
//		}
//	}
//	for (int i = 0; i < SZ1; i++) {
//		for (int j = 0; j < SZ2; j++) {
//			W2[i][j] = W2[i][j] - learning_rate * W2_gradient[i][j];
//		}
//	}
//	for (int i = 0; i < SZ0; i++) {
//		for (int j = 0; j < SZ1; j++) {
//			W1[i][j] = W1[i][j] - learning_rate * W1_gradient[i][j];;
//		}
//	}
//
//	for (int i = 0; i < SZ3; i++) {
//		B3[i] = B3[i] - learning_rate * B3_gradient[i];
//	}
//	for (int i = 0; i < SZ2; i++) {
//		B2[i] = B2[i] - learning_rate * B2_gradient[i];;
//	}
//	for (int i = 0; i < SZ1; i++) {
//		B1[i] = B1[i] - learning_rate * B1_gradient[i];;
//	}
//}
//
//void Accumulate_gradients_of_parameters(int testCase) {
//	for (int i = 0; i < SZ2; i++) {
//		for (int j = 0; j < SZ3; j++) {
//			W3_gradient[i][j] += Delta3[j] * F2[i];
//		}
//	}
//	for (int i = 0; i < SZ1; i++) {
//		for (int j = 0; j < SZ2; j++) {
//			W2_gradient[i][j] += Delta2[j] * F1[i];
//		}
//	}
//	for (int i = 0; i < SZ0; i++) {
//		for (int j = 0; j < SZ1; j++) {
//			W1_gradient[i][j] += Delta1[j] * F0[i];
//		}
//	}
//
//	for (int i = 0; i < SZ3; i++) {
//		B3_gradient[i] += Delta3[i];
//	}
//	for (int i = 0; i < SZ2; i++) {
//		B2_gradient[i] += Delta2[i];
//	}
//	for (int i = 0; i < SZ1; i++) {
//		B1_gradient[i] += Delta1[i];
//	}
//}
//
//void Do_train() {
//	int idx_miniBatch, start_of_this_minibatch, start_of_next_minibatch,count =0;
//	for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
//		idx_miniBatch = 0;
//		do {
//			start_of_this_minibatch = idx_miniBatch * SIZE_MINIBATCH;
//			start_of_next_minibatch = start_of_this_minibatch + SIZE_MINIBATCH;
//			if (!(start_of_next_minibatch - 1 <= TRAIN_SIZE - 1)) {
//				break;
//			}
//
//			initialize_gradients_before_minibatch();
//			for (int index = start_of_this_minibatch; index < start_of_next_minibatch; index++) {
//        if (index % 2000 == 1) {
//          printf("Train Supervised NN: At epoch %d. learning_rate = %f, Training example number = %d\n", epoch, learning_rate, index);
//        }
//        forward_compute(index);
//				backward_compute(index);
//				Accumulate_gradients_of_parameters(index);
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
//
//	for (int index = 0; index < TEST_SIZE; index++) {
//    if (index % 1000 == 0) {
//      printf("Number of test examples tested = %d\n", index);
//    }
//
//		for (int row = 0; row < ROW; row++) {
//			for (int column = 0; column < COLUMN; column++) {
//				F0[row * 28 + column] = Test_Data[index][row][column] / 255;
//			}
//		}
//
//		for (int i = 0; i < SZ1; i++) {
//			F1[i] = Compute_sigmoid_F1_from_F0(i);
//		}
//		for (int i = 0; i < SZ2; i++) {
//			F2[i] = Compute_sigmoid_F2_from_F1(i);
//		}
//		for (int i = 0; i < SZ3; i++) {
//			F3[i] = Compute_IN3_from_F2(i);
//		}
//
//		temp = 0.0;
//		for (int i = 0; i < SZ3; i++) {
//			temp = temp + exp(F3[i]);
//		}
//
//		for (int i = 0; i < SZ3; i++) {
//			SoftMax[i] = exp(F3[i]) / temp;
//		}
//
//		int max_idx = 0;
//		double max_val = SoftMax[0];
//		for (int i = 1; i < SZ3; i++) {
//			if (max_val < SoftMax[i]) {
//				max_idx = i;
//				max_val = SoftMax[i];
//			}
//		}
//
//		int Label_by_model = max_idx;    
//		if (Label_by_model == Test_Label[index])
//			correct_cnt++;
//	}
//	double accuracy = ((double)correct_cnt) / TEST_SIZE;
//	return accuracy;
//}
//
//double Compute_sigmoid_F1_from_F0(int i) {
//	double res, x = 0;
//
//  for (int j = 0; j < SZ0; j++) {
//    x += W1[j][i] * F0[j];
//  }
//	x += B1[i];
//	IN1[i] = x;
//	res = 1.0 / (1 + exp(-x));
//	return res;
//}
//
//double Compute_sigmoid_F2_from_F1(int i) {
//	double res, x = 0;
//
//  for (int j = 0; j < SZ1; j++) {
//    x += W2[j][i] * F1[j];
//  }
//	x += B2[i];
//	IN2[i] = x;
//	res = 1.0 / (1 + exp(-x));
//	return res;
//}
//
//double Compute_IN3_from_F2(int i) {
//	double res, x = 0;
//
//  for (int j = 0; j < SZ2; j++) {
//    x += W3[j][i] * F2[j];
//  }
//	x += B3[i];
//	IN1[i] = x;
//	return x;
//}