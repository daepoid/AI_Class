//#define _CRT_SECURE_NO_WARNINGS
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include<string.h>
//#include<ctype.h>
//#include<math.h>
//
////Constants
////Hyper Parameters
//#define Sz0									784	//visible layer 갯수
//#define Sz1									500 //hidden layer 1 갯수
//#define Sz2									400	//hidden layer 2 갯수
//#define Sz3									10 //output layer 갯수
//#define Ori_learning_rate					0.05 //오차의 수정을 어느정도 반영할지 결정하는 학습비율
//
//#define MAX_epoch							20 // epoch 횟수
//#define Num_train							60000 // training 횟수
//#define Num_test							10000 // test 횟수
//#define Size_minibatch						4 //mini batch size
//
////Neuon output variables of all layers
//double F0[Sz0];			//sum of visible layer
//double F1[Sz1];			//sum of hidden 1
//double F2[Sz2];			//sum of 2
//double F3[Sz3];			//sum of 3(final layers)
//
////Weighted sum varaibles
//double IN1[Sz0];		//sum of weighted input to units on layer1 s= b + ∑w(i)f(i) (w(i) : i번째 weight f(i) : i번째 출력값)
//double IN2[Sz1];		//sum of weighted input to units on layer2 s= b + ∑w(i)f(i) (w(i) : i번째 weight f(i) : i번째 출력값)
//double IN3[Sz2];		//sum of weighted input to units on layer3 s= b + ∑w(i)f(i) (w(i) : i번째 weight f(i) : i번째 출력값)
//
////Accumulation storage for gradients
//double W1_gradient[Sz0][Sz1];
//double W2_gradient[Sz1][Sz2];
//double W3_gradient[Sz2][Sz3];
//double B1_gradient[Sz1];
//double B2_gradient[Sz2];
//double B3_gradient[Sz3];
//
////Parameters
//double W1[Sz0][Sz1];	//sum of Weight matrix 1
//double W2[Sz1][Sz2];	//sum of Weight matrix 2
//double W3[Sz2][Sz3];	//sum of Weight matrix 3
//double B1[Sz1];			//bias of layer1
//double B2[Sz2];			//bias of layer2
//double B3[Sz3];			//bias of layer3
//double SoftMax[Sz3];	//최종단계에서 f3의 index(0~9)값 확률을 저장한 배열
//
////Deltas
//double Delta1[Sz1];		//layer 1 오차
//double Delta2[Sz2];		//layer 2 오차
//double Delta3[Sz3];		//layer 3 오차..?
//
//double sigmoid_deriv1[Sz1];
//double sigmoid_deriv2[Sz2];
//
//double Train_Data[Num_train][28][28];	//Original data vrought from the file
//int Trian_Label_ori[Num_train];		//Original data vrought from the file
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
//		if (leng != 2) { //왜 leng이 2여야만하는지..?? -> \n이 포함되있음
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
//			}//end c for loop
//		}//end r for loop
//	}//end tastCase for loop
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
//		if (leng != 2) { //왜 leng이 2여야만하는지..?? -> \n이 포함되있음
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
//			}//end c for loop
//		}//end r for loop
//	}//end tastCase for loop
//	printf("Test data loading finish.\n");
//}
//
//void Init_Parameter() {//Weight 하고 bias를 랜덤값으로 초기화
//	//initialize W1
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1[i][j] = getRandNum();
//		}
//	}
//
//	//initialize W2
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2[i][j] = getRandNum();
//		}
//	}
//	//initialize W3
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3[i][j] = getRandNum();
//		}
//	}
//	//initialize Bias of layer3
//	for (int i = 0; i < Sz3; i++) {
//		B3[i] = getRandNum();
//	}
//	//initialize Bias of layer2
//	for (int i = 0; i < Sz2; i++) {
//		B2[i] = getRandNum();
//	}
//	//initialize Bias of layer1
//	for (int i = 0; i < Sz1; i++) {
//		B1[i] = getRandNum();
//	}
//}
//
//void initialize_gradients_before_minibatch() {
//	//initialize W1_gradient
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1_gradient[i][j] = 0;
//		}
//	}
//
//	//initialize W2_gradient
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2_gradient[i][j] = 0;
//		}
//	}
//	//initialize W3_gradient
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3_gradient[i][j] = 0;
//		}
//	}
//	//initialize Bias_gradient of layer3
//	for (int i = 0; i < Sz3; i++) {
//		B3_gradient[i] = 0;
//	}
//	//initialize Bias_gradient of layer2
//	for (int i = 0; i < Sz2; i++) {
//		B2_gradient[i] = 0;
//	}
//	//initialize Bias_gradient of layer1
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
//
//	if (r2 < RAND_MAX / 2)
//		sign = -1.0;
//	else
//		sign = 1.0;
//
//	return (r / (double)RAND_MAX) * sign;
//}
//
//void forward_compute(int testCase) {
//	//start Forward Computation
//	//provide a training example to layer 0
//	int j = 0;
//	for (int r = 0; r < 28; r++) {
//		for (int c = 0; c < 28; c++) {
//			F0[j] = Train_Data[testCase][r][c] / 255; // 0~1사이의 값으로 조정
//			j++;
//		}//end c for loop
//	}//end r for loop
//
//	//compute the output of units of layer 1.
//	for (int i = 0; i < Sz1; i++) {
//		F1[i] = Compute_sigmoid_F1_from_F0(i);
//		// back propagation에서 사용할 미분함수를 미리 구함 f'(x) = f(x)*(1-f(x))
//		sigmoid_deriv1[i] = F1[i] * (1 - F1[i]);
//	}
//
//	//compute the output of units of layer 1.
//	for (int i = 0; i < Sz2; i++) {
//		F2[i] = Compute_sigmoid_F2_from_F1(i);
//		// back propagation에서 사용할 미분함수를 미리 구함 f'(x) = f(x)*(1-f(x))
//		sigmoid_deriv2[i] = F2[i] * (1 - F2[i]);
//	}
//
//	//compute the output of units of layer 3 this is linear operation.
//	for (int i = 0; i < Sz3; i++) {
//		F3[i] = Compute_IN3_from_F2(i);
//	}
//	double denomi;
//	//compute sum of exponential of all output of final layer to use for denominator
//	denomi = 0.0;
//	for (int i = 0; i < Sz3; i++) {
//		denomi = denomi + exp(F3[i]);
//	}
//
//	//compute softMax for F3 units.
//	//SoftMax[i] = e^f3(i)/∑e^f3(i)
//	for (int i = 0; i < Sz3; i++) {
//		SoftMax[i]= exp(F3[i]) / denomi;
//	}
//	//end Forward Computation
//
//}
//
//void backward_compute(int testCase) {
//	int correct_label;
//	//Backward propagation start
//
//	//output layer delta 계산
//	//Delta3 = e^f3(i)/∑e^f3(i)(SoftMax랑 같음)
//
//	for (int i = 0; i < Sz3; i++) {
//		Delta3[i] = SoftMax[i];
//	}
//
//	// if i=j 
//	correct_label = Trian_Label_ori[testCase];
//	Delta3[correct_label] = Delta3[correct_label] - 1;
//
//
//	//hidden layer 2 delta 계산
//	//Delta2 = sigmoid' * ∑Delta3*W3
//
//	//compute delta for F2 units.
//	for (int j = 0; j < Sz2; j++) {
//		double temp = 0;
//		for (int k = 0; k < Sz3; k++) {
//			temp = temp + Delta3[k] * W3[j][k];
//		}
//		Delta2[j] = sigmoid_deriv2[j] * temp;
//	}
//
//	//hidden layer 1 delta 계산
//	//Delta1 = sigmoid' * ∑Delta2*W2
//
//	//compute delta for F1 units.
//	for (int j = 0; j < Sz1; j++) {
//		double temp = 0;
//		for (int k = 0; k < Sz2; k++) {
//			temp = temp + Delta2[k] * W2[j][k];
//		}
//		Delta1[j] = sigmoid_deriv1[j] * temp;
//	}
//	//end back propagation
//
//}
//
//void update_parameters_for_minibatch() {
//	//Update parameters
//
//	//update W3 parameters
//	//W3 = W3 - ∑learning_rate * Delta3 * F2
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3[i][j] = W3[i][j] - learning_rate * W3_gradient[i][j];
//		}
//	}
//
//	//update W2 parameters
//	//W2 = W2 - ∑learning_rate * Delta2 * F1
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2[i][j] = W2[i][j] - learning_rate * W2_gradient[i][j];
//		}
//	}
//
//	//update W1 parameters
//	//W1 = W1 - ∑learning_rate * Delta1 * F0
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1[i][j] = W1[i][j] - learning_rate * W1_gradient[i][j];;
//		}
//	}
//
//	//update B3 parameters
//	// B3 = B3 - ∑learning_rate * Delta3
//	for (int i = 0; i < Sz3; i++) {
//		B3[i] = B3[i] - learning_rate * B3_gradient[i];
//	}
//
//	//update B2 parameters
//	// B2 = B2 - ∑learning_rate * Delta2
//	for (int i = 0; i < Sz2; i++) {
//		B2[i] = B2[i] - learning_rate * B2_gradient[i];;
//	}
//
//	//update B1 parameters
//	// B1 = B1 - ∑learning_rate * Delta1
//	for (int i = 0; i < Sz1; i++) {
//		B1[i] = B1[i] - learning_rate * B1_gradient[i];;
//	}
//	//Update parameters end
//}
//
//void Accumulate_gradients_of_parameters(int testCase) {
//	//Start accumulation Gradient
//
//	//update W3 parameters
//	//W3 = W3 - ∑learning_rate * Delta3 * F2
//	for (int i = 0; i < Sz2; i++) {
//		for (int j = 0; j < Sz3; j++) {
//			W3_gradient[i][j] += Delta3[j] * F2[i];
//		}
//	}
//
//	//update W2 parameters
//	//W2 = W2 - ∑learning_rate * Delta2 * F1
//	for (int i = 0; i < Sz1; i++) {
//		for (int j = 0; j < Sz2; j++) {
//			W2_gradient[i][j] += Delta2[j] * F1[i];
//		}
//	}
//
//	//update W1 parameters
//	//W1 = W1 - ∑learning_rate * Delta1 * F0
//	for (int i = 0; i < Sz0; i++) {
//		for (int j = 0; j < Sz1; j++) {
//			W1_gradient[i][j] += Delta1[j] * F0[i];
//		}
//	}
//
//	//update B3 parameters
//	// B3 = B3 - ∑learning_rate * Delta3
//	for (int i = 0; i < Sz3; i++) {
//		B3_gradient[i] += Delta3[i];
//	}
//
//	//update B2 parameters
//	// B2 = B2 - ∑learning_rate * Delta2
//	for (int i = 0; i < Sz2; i++) {
//		B2_gradient[i] += Delta2[i];
//	}
//
//	//update B1 parameters
//	// B1 = B1 - ∑learning_rate * Delta1
//	for (int i = 0; i < Sz1; i++) {
//		B1_gradient[i] += Delta1[i];
//	}
//	//end accumulation Gradient
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
//	}//end epoch for loop
//}
//
//double Do_test() {
//	double temp;
//	int correct_cnt = 0;
//	for (int testCase = 0; testCase < Num_test; testCase++) {
//		if (testCase % 1000 == 0)
//			printf("Test example Number : %d\n", testCase);
//
//		//provide a training example to layer 0
//		for (int r = 0; r < 28; r++) {
//			for (int c = 0; c < 28; c++) {
//				F0[r * 28 + c] = Test_Data[testCase][r][c] / 255;
//			}//end c for loop
//		}//end r for loop
//
//		//compute the output of units of layer 1.
//		for (int i = 0; i < Sz1; i++) {
//			F1[i] = Compute_sigmoid_F1_from_F0(i);
//		}
//
//		//compute the output of units of layer 1.
//		for (int i = 0; i < Sz2; i++) {
//			F2[i] = Compute_sigmoid_F2_from_F1(i);
//		}
//
//		//compute the output of units of layer 3 this is linear operation.
//		for (int i = 0; i < Sz3; i++) {
//			F3[i] = Compute_IN3_from_F2(i);
//		}
//
//		//softMax is to be compute.
//		temp = 0.0;
//		for (int i = 0; i < Sz3; i++) {
//			temp = temp + exp(F3[i]);
//		}
//
//		//Backward pass start
//
//		//SoftMax is finally obtained
//		for (int i = 0; i < Sz3; i++) {
//			SoftMax[i] = exp(F3[i]) / temp;
//		}
//
//		//Find the label determined by the model.
//		int max_idx = 0;
//		double max_val = SoftMax[0];
//		for (int i = 1; i < Sz3; i++) {
//			if (max_val < SoftMax[i]) {
//				max_idx = i;
//				max_val = SoftMax[i];
//			}
//		}
//		//Compare it with the gold-standard label.
//		int Label_by_model = max_idx;
//
//		if (Label_by_model == Test_Label[testCase])
//			correct_cnt++;
//	}//end testCase for loop
//	double accuracy = ((double)correct_cnt) / Num_test;
//	return accuracy;
//}
//
//double Compute_sigmoid_F1_from_F0(int i) {
//	//visible layer 0의 forward computation 계산
//	//x = B1 + ∑ W1F0
//	//simoid(x) -> 출력
//
//	double res, x = 0;
//
//	for (int j = 0; j < Sz0; j++)
//		x += W1[j][i] * F0[j];
//	x += B1[i];
//	IN1[i] = x;
//	res = 1.0 / (1 + exp(-x)); // sigmoid func ( 1/(1+e^(-x) )
//	return res;
//}
//
//double Compute_sigmoid_F2_from_F1(int i) {
//	//hidden layer 1의 forward computation 계산
//	//x = B2 + ∑ W2F1
//	//simoid(x) -> 출력
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
//	//hidden layer 2의 forward computation 계산
//	//x = B3 + ∑ W3F2
//	//simoid(x) -> 출력 -> 마지막 층은 안함..? 왜??
//	double res, x = 0;
//
//	for (int j = 0; j < Sz2; j++)
//		x += W3[j][i] * F2[j];
//	x += B3[i];
//	IN1[i] = x;
//	return x;
//}