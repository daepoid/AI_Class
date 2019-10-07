//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include <math.h>
//#include <ctype.h>
//#include <string.h>
//#pragma warning(disable: 4996)
//
///*
//  배치 정규화 알고리즘 사용하기
//  data와 label을 pair나 map으로 묶어서 사용하는 경우?
//  learning rate, layer, random_state, epoch를 쉽게 조정할 수 있게 함수로 만들고 외부에서 처리할 수 있게 하기
//  cross_validation, rmse 구현
//  grid_search 구현 가능?
//*/
//
//#define Sz0 784
//#define Sz1 500
//#define Sz2 400
//#define Sz3 10
//
//#define MAX_EPOCH 20
//#define TRAIN_SIZE 60000
//#define TEST_SIZE 10000
//#define ori_learning_rate 0.08
//#define SIZE_MINIBATCH 4
//
//#define ROW 28
//#define COLUMN 28
//
//double F0[Sz0];
//double F1[Sz1];
//double F2[Sz2];
//double F3[Sz3];
//double SoftMax[Sz3];
//
//double W1[Sz0][Sz1];
//double W2[Sz1][Sz2];
//double W3[Sz2][Sz3];
//double B1[Sz1];
//double B2[Sz2];
//double B3[Sz3];
//
//// TODO : check
//double W1_gradient[Sz0][Sz1];
//double W2_gradient[Sz1][Sz2];
//double W3_gradient[Sz2][Sz3];
//double B1_gradient[Sz1];
//double B2_gradient[Sz2];
//double B3_gradient[Sz3];
//
//double IN1[Sz1];
//double IN2[Sz2];
//double IN3[Sz3];
//
//double Delta1[Sz1];
//double Delta2[Sz2];
//double Delta3[Sz3];
//
//double Sigmoid_derivative1[Sz1];
//double Sigmoid_derivative2[Sz2];
//
//int Train_Data[TRAIN_SIZE][ROW][COLUMN];
//int Train_Label[TRAIN_SIZE];
//
//double learning_rate = ori_learning_rate;
//
//int Test_Data[TEST_SIZE][ROW][COLUMN];
//int Test_Label[TEST_SIZE];
//
//// TODO : 밑에 전역변수들 수정하기
//
//
//FILE* fp_accuracy;
//
//void Read_training_data();
//void Read_testing_data();
//void init_Parameters();
//void Do_train(); 
//double Do_test();
//
//void forward_compute(int i);
//void backward_compute(int i);
//void Accumulate_gradients_of_parameters(int i);
//void initialize_gradients_before_minibatch();
//void update_parameters_for_minibatch();
//
//double getRandNum();
//double Compute_sigmoid_F1_from_F0(int i);
//double Compute_sigmoid_F2_from_F1(int i);
//double Compute_IN3_from_F2(int i);
//
//
//int main() {
//  fp_accuracy = fopen("Test_Accuracy_Batch.txt", "a");
//  if (!fp_accuracy) {
//    printf("File Open Failed : Test_Accuracy_Batch.txt\n");
//    return 0;
//  }
//
//  srand(time(NULL)); // random_state를 고정하고 하이퍼 파라미터와 함께 저장하도록 바꾸기
//
//  fprintf(fp_accuracy, "\n\nExperiment of building Supervised NN system has started.\n");
//  fprintf(fp_accuracy, "sz0 = %5d, sz1 = %5d, sz2 = %5d, sz3 = %5d, train_set_size = %6d, test_set_size = %6d, learning_rate = %lf\n",
//    Sz0, Sz1, Sz2, Sz3, TRAIN_SIZE, TEST_SIZE, learning_rate);
//  // 하이퍼 파라미터를 파일에 입력
//  fflush(fp_accuracy);
//
//
//  printf("Read Training Data\n");
//  Read_training_data();
//
//  printf("Read Testing Data\n");
//  Read_testing_data();
//
//  printf("Init Parameters\n");
//  init_Parameters();
//
//  printf("Do Train Batch Ver 1.0 \n");
//  Do_train();
//
//  fclose(fp_accuracy);
//  return 1;
//}
//
//void Read_training_data() {
//  FILE* fp;
//  char* ret_fgets;
//  char buff_line[1024];
//  char num_str[20];
//  int leng;
//
//  fp = fopen("datasets/train.txt", "r");
//  if (!fp) {
//    printf("File Open Failed : train.txt\n");
//    return;
//  }
//
//  for (int index = 0; index < TRAIN_SIZE; index++) {
//    if (index % 10000 == 0) {
//      printf("Train data loading. train_index = %d\n", index);
//    }
//    ret_fgets = fgets(buff_line, 1024, fp);
//    if (!ret_fgets) {
//      printf("fgets error1.\n");
//      return;
//    }
//    leng = strlen(buff_line);
//    if (leng != 2) {
//      printf("wrong leng of error.\n");
//      getchar();
//    }
//    buff_line[--leng] = '\0';
//    Train_Label[index] = atoi(buff_line);
//
//    for (int row = 0; row < ROW; row++) {
//      ret_fgets = fgets(buff_line, 1024, fp);
//      if (!ret_fgets) {
//        printf("fgets error2.\n");
//        return;
//      }
//
//      leng = strlen(buff_line);
//      buff_line[--leng] = '\0';
//
//      int cp = 0;
//      for (int column = 0; column < COLUMN; column++) {
//        while (buff_line[cp] == ' ')
//          cp++;
//        int i = 0;
//        while (isdigit(buff_line[cp])) {
//          num_str[i] = buff_line[cp];
//          i++;
//          cp++;
//        }
//        num_str[i] = '\0';
//        Train_Data[index][row][column] = atof(num_str);
//      }
//    }
//  }
//  //fclose(fp);
//  /*fp = fopen("datasets/test.txt", "r");
//  if (!fp) {
//    printf("File open failed : test.txt\n");
//    return;
//  }
//
//  for (index = 0; index < test_set_size; index++) {
//    if (index % 2000 == 0) {
//      printf("Test data loading. test_index = %d\n", index);
//    }
//    fscanf(fp, "%d", &test_label[index]);
//
//    for (row = 0; row < ROW; row++) {
//      for (column = 0; column < COLUMN; column++) {
//        fscanf(fp, "%d", &temp);
//        test_data[index][row][column] = temp;
//      }
//    }
//  }
//  fclose(fp);*/
//}
//
//void Read_testing_data() {
//  FILE* fp;
//  char* ret_fgets;
//  char buff_line[1024];
//  char num_str[20];
//  int leng;
//
//  fp = fopen("datasets/test.txt", "r");
//  if (!fp) {
//    printf("File open failed : test.txt\n");
//    return;
//  }
//
//  for (int index = 0; index < TEST_SIZE; index++) {
//    if (index % 2000 == 0) {
//      printf("Test data loading. test_index = %d\n", index);
//    }
//    ret_fgets = fgets(buff_line, 1024, fp);
//    if (!ret_fgets) {
//      printf("fgets error1.\n");
//      return;
//    }
//    leng = strlen(buff_line);
//    if (leng != 2) {
//      printf("wrong leng of error.\n");
//      getchar();
//    }
//    buff_line[--leng] = '\0';
//    Test_Label[index] = atoi(buff_line);
//
//    for (int row = 0; row < ROW; row++) {
//      ret_fgets = fgets(buff_line, 1024, fp);
//      if (!ret_fgets) {
//        printf("fgets error1.\n");
//        return;
//      }
//
//      leng = strlen(buff_line);
//      buff_line[--leng] = '\0';
//
//      int cp = 0;
//      for (int column = 0; column < COLUMN; column++) {
//        while (buff_line[cp] == ' ')
//          cp++;
//        int i = 0;
//        while (isdigit(buff_line[cp])) {
//          num_str[i] = buff_line[cp];
//          i++;
//          cp++;
//        }
//        num_str[i] = '\0';
//        Test_Data[index][row][column] = atof(num_str);
//      }
//    }
//  }
//  //fclose(fp);
//}
//
//void init_Parameters() {
//  for (int i = 0; i < Sz0; i++) {
//    for (int j = 0; j < Sz1; j++) {
//      W1[i][j] = getRandNum();
//    }
//  }
//  for (int i = 0; i < Sz1; i++) {
//    for (int j = 0; j < Sz2; j++) {
//      W2[i][j] = getRandNum();
//    }
//  }
//  for (int i = 0; i < Sz2; i++) {
//    for (int j = 0; j < Sz3; j++) {
//      W3[i][j] = getRandNum();
//    }
//  }
//
//  for (int i = 0; i < Sz3; i++) {
//    B3[i] = getRandNum();
//  }
//  for (int i = 0; i < Sz2; i++) {
//    B2[i] = getRandNum();
//  }
//  for (int i = 0; i < Sz1; i++) {
//    B1[i] = getRandNum();
//  }
//}
//
//void initialize_gradients_before_minibatch() {
//  for (int i = 0; i < Sz0; i++) {
//    for (int j = 0; j < Sz1; j++) {
//      W1_gradient[i][j] = 0;
//    }
//  }
//  for (int i = 0; i < Sz1; i++) {
//    for (int j = 0; j < Sz2; j++) {
//      W2_gradient[i][j] = 0;
//    }
//  }
//  for (int i = 0; i < Sz2; i++) {
//    for (int j = 0; j < Sz3; j++) {
//      W3_gradient[i][j] = 0;
//    }
//  }
//
//  for (int i = 0; i < Sz3; i++) {
//    B3_gradient[i] = 0;
//  }
//  for (int i = 0; i < Sz2; i++) {
//    B2_gradient[i] = 0;
//  }
//  for (int i = 0; i < Sz1; i++) {
//    B1_gradient[i] = 0;
//  }
//}
//
//void forward_compute(int index) {
//
//  int j = 0;
//  for (int row = 0; row < ROW; row++) {
//    for (int column = 0; column < COLUMN; column++) {
//      F0[j] = Train_Data[index][row][column] / 255;
//      j++;
//    }
//  }
//
//  for (int i = 0; i < Sz1; i++) {
//    F1[i] = Compute_sigmoid_F1_from_F0(i);
//    Sigmoid_derivative1[i] = F1[i] * (1 - F1[i]);
//  }
//  for (int i = 0; i < Sz2; i++) {
//    F2[i] = Compute_sigmoid_F2_from_F1(i);
//    Sigmoid_derivative2[i] = F2[i] * (1 - F2[i]);
//  }
//  for (int i = 0; i < Sz3; i++) {
//    F3[i] = Compute_IN3_from_F2(i);
//  }
//
//  double denomi = 0.0;
//  for (int i = 0; i < Sz3; i++) {
//    denomi = denomi + exp(F3[i]);
//  }
//  for (int i = 0; i < Sz3; i++) {
//    SoftMax[i] = exp(F3[i]) / denomi;
//  }
//}
//
//void backward_compute(int te) {
//  int correct_label;
//  double temp;
//
//  // foward와 같이 계산?
//  for (int i = 0; i < Sz3; i++) {
//    Delta3[i] = SoftMax[i];
//  }
//
//  correct_label = Train_Label[te];
//  Delta3[correct_label] = Delta3[correct_label] - 1;
//
//  for (int j = 0; j < Sz2; j++) {
//    temp = 0;
//    for (int k = 0; k < Sz3; k++) {
//      temp = temp + Delta3[k] * W3[j][k];
//    }
//    Delta2[j] = Sigmoid_derivative2[j] * temp;
//  }
//
//  for (int j = 0; j < Sz1; j++) {
//    temp = 0;
//    for (int k = 0; k < Sz2; k++) {
//      temp = temp + Delta2[k] * W2[j][k];
//    }
//    Delta1[j] = Sigmoid_derivative1[j] * temp;
//  }
//}
//
//void update_parameters_for_minibatch() {
//  for (int i = 0; i < Sz2; i++) {
//    for (int j = 0; j < Sz3; j++) {
//      W3[i][j] = W3[i][j] - learning_rate * W3_gradient[i][j];
//    }
//  }
//  for (int i = 0; i < Sz1; i++) {
//    for (int j = 0; j < Sz2; j++) {
//      W2[i][j] = W2[i][j] - learning_rate * W2_gradient[i][j];
//    }
//  }
//  for (int i = 0; i < Sz0; i++) {
//    for (int j = 0; j < Sz1; j++) {
//      W1[i][j] = W1[i][j] - learning_rate * W1_gradient[i][j];;
//    }
//  }
//
//  for (int i = 0; i < Sz3; i++) {
//    B3[i] = B3[i] - learning_rate * B3_gradient[i];
//  }
//  for (int i = 0; i < Sz2; i++) {
//    B2[i] = B2[i] - learning_rate * B2_gradient[i];;
//  }
//  for (int i = 0; i < Sz1; i++) {
//    B1[i] = B1[i] - learning_rate * B1_gradient[i];;
//  }
//}
//
//void Accumulate_gradients_of_parameters(int index) {
//  for (int i = 0; i < Sz2; i++) {
//    for (int j = 0; j < Sz3; j++) {
//      W3_gradient[i][j] += Delta3[j] * F2[i];
//    }
//  }
//  for (int i = 0; i < Sz1; i++) {
//    for (int j = 0; j < Sz2; j++) {
//      W2_gradient[i][j] += Delta2[j] * F1[i];
//    }
//  }
//  for (int i = 0; i < Sz0; i++) {
//    for (int j = 0; j < Sz1; j++) {
//      W1_gradient[i][j] += Delta1[j] * F0[i];
//    }
//  }
//
//  for (int i = 0; i < Sz3; i++) {
//    B3_gradient[i] += Delta3[i];
//  }
//  for (int i = 0; i < Sz2; i++) {
//    B2_gradient[i] += Delta2[i];
//  }
//  for (int i = 0; i < Sz1; i++) {
//    B1_gradient[i] += Delta1[i];
//  }
//}
//
//void Do_train() {
//  int index_minibatch;
//  int start_of_this_minibatch, start_of_next_minibatch;
//  double Accuracy;
//
//  for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
//    index_minibatch = 0;
//    do {
//      start_of_this_minibatch = index_minibatch * SIZE_MINIBATCH;
//      start_of_next_minibatch = start_of_this_minibatch + SIZE_MINIBATCH;
//      if (!(start_of_next_minibatch - 1 <= TRAIN_SIZE - 1)) {
//        break;
//      }
//      initialize_gradients_before_minibatch();
//      for (int index = start_of_this_minibatch; index < start_of_next_minibatch; index++) {
//        if (index % 2000 == 1) {
//          printf("Train Supervised NN: At epoch %d. learning_rate = %f, Training example number = %d\n", epoch, learning_rate, index);
//        }
//        forward_compute(index);
//        backward_compute(index);
//        Accumulate_gradients_of_parameters(index);
//      }
//      update_parameters_for_minibatch();
//      index_minibatch += 1;
//    } while (1);
//
//    printf("\nSupervised NN system Testing after epoch = %d has started.\n", epoch);
//    double Accuracy = Do_test();
//    fprintf(fp_accuracy, "After epoch = %d, learning_rate = %f, S-NN Accuracy = %f\n", epoch, learning_rate, Accuracy);
//    fflush(fp_accuracy);
//    printf("S-NN Accuracy after epoch %3d = %f (stored in file Test_Accuracy.txt)\n\n", epoch, Accuracy);
//  }
//}
//
//double Do_test() {
//  int correct_count = 0;
//  double temp;
//
//  for (int index = 0; index < TEST_SIZE; index++) {
//    if (index % 1000 == 0) {
//      printf("Number of test examples tested = %d\n", index);
//    }
//
//    for (int row = 0; row < ROW; row++) {
//      for (int column = 0; column < COLUMN; column++) {
//        F0[row * 28 + column] = Test_Data[index][row][column] / 255;
//      }
//    }
//
//    for (int i = 0; i < Sz1; i++) {
//      F1[i] = Compute_sigmoid_F1_from_F0(i);
//    }
//    for (int i = 0; i < Sz2; i++) {
//      F2[i] = Compute_sigmoid_F2_from_F1(i);
//    }
//    for (int i = 0; i < Sz3; i++) {
//      F3[i] = Compute_IN3_from_F2(i);
//    }
//
//    temp = 0.0;
//    for (int i = 0; i < Sz3; i++) {
//      temp = temp + exp(F3[i]);
//    }
//
//    for (int i = 0; i < Sz3; i++) {
//      SoftMax[i] = exp(F3[i]) / temp;
//    }
//
//    int max_idx = 0;
//    double max_val = SoftMax[0];
//    for (int i = 1; i < Sz3; i++) {
//      if (max_val < SoftMax[i]) {
//        max_idx = i;
//        max_val = SoftMax[i];
//      }
//    }
//
//    int label_by_model = max_idx;
//    if (label_by_model == Test_Label[index]) {
//      correct_count++;
//    }
//  }
//  double accuracy = ((double)correct_count) / TEST_SIZE;
//  return accuracy;
//}
//
//double getRandNum() {
//  double r, sign;
//  int r2;
//
//  r = (double)rand();
//  r2 = rand();
//
//  if (r2 < RAND_MAX / 2)
//    sign = -1.0;
//  else
//    sign = 1.0;
//
//  return (r / (double)RAND_MAX) * sign;
//}
//
//double Compute_sigmoid_F1_from_F0(int i) {
//  double res, x = 0;
//  for (int j = 0; j < Sz0; j++) {
//    x += W1[j][i] * F0[j];
//  }
//  x += B1[i];
//  IN1[i] = x;
//  res = 1.0 / (1 + exp(-x));
//  return res;
//}
//
//double Compute_sigmoid_F2_from_F1(int i) {
//  double res, x = 0;
//  for (int j = 0; j < Sz1; j++) {
//    x += W2[j][i] * F1[j];
//  }
//  x += B2[i];
//  IN2[i] = x;
//  res = 1.0 / (1 + exp(-x));
//  return res;
//}
//
//double Compute_IN3_from_F2(int i) {
//  double x = 0;
//  for (int j = 0; j < Sz2; j++) {
//    x += W3[j][i] * F2[j];
//  }
//  x += B3[i];
//  IN3[i] = x;
//  return x;
//}
