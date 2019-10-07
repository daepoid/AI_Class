//#include<iostream>
//#include<cstdlib>
//#include<cmath>
//#include<ctime>
//using namespace std;
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
//#define TRAIN_SIZE 60000
//#define TEST_SIZE 10000
//#define ori_learning_rate 0.05
//#define MAX_EPOCH 100
//#define ROW 28
//#define COLUMN 28
//
//#define SIZE_MINIBATCH 32;
//
//// training and test data
//double learning_rate = ori_learning_rate;
//
//int Train_Data[60000][28][28];
//int Train_Label[60000];
//
//int Test_Data[10000][28][28];
//int Test_Label[10000];
//
//// weight and bias parameters
//const int sz0 = 784;
//const int sz1 = 500;
//const int sz2 = 400;
//const int sz3 = 10;
//
//double W1[sz0][sz1];
//double W2[sz1][sz2];
//double W3[sz2][sz3];
//
//double B1[sz1];
//double B2[sz2];
//double B3[sz3];
//
//// output of neurons
//double F0[sz0];
//double F1[sz1];
//double F2[sz2];
//double F3[sz3];
//double SoftMax[sz3];
//
//// TODO : 밑에 전역변수들 수정하기
//// temp
//double Sigmoid_deriv1[sz1];
//double Sigmoid_deriv2[sz2];
//
//// Weighted sums
//double IN1[sz1];
//double IN2[sz2];
//double IN3[sz3];
//
//// Delta variables
//double Delta1[sz1];
//double Delta2[sz2];
//double Delta3[sz3];
//
//// Batch
//double W1_gradient[sz0][sz1];
//double W2_gradient[sz1][sz2];
//double W3_gradient[sz2][sz3];
//
//double B1_gradient[sz1];
//double B2_gradient[sz2];
//double B3_gradient[sz3];
//
//FILE* fp_accuracy;
//
//void Read_training_data();
//void init_Parameters();
//double getRandNum();
//void Do_train();
//double Compute_sigmoid_F1_from_F0(int i);
//double Compute_sigmoid_F2_from_F1(int i);
//double Compute_IN3_from_F2(int i);
//double Do_test();
//
//void initialize_gradients_before_minibatch();
//void forward_compute(int i);
//void backward_compute(int i);
//void Accumulate_gradients_of_parameters(int i);
//void update_parameters_for_minibatch();
//
//int main() {
//  fp_accuracy = fopen("Test_Accuracy.txt", "a");
//  if (!fp_accuracy) {
//    printf("File Open Failed : Test_Accuracy.txt\n");
//    return 0;
//  }
//  fprintf(fp_accuracy, "\n\nExperiment of building Supervised NN system has started.\n");
//  fprintf(fp_accuracy, "sz0 = %5d, sz1 = %5d, sz2 = %5d, sz3 = %5d, train_set_size = %6d, test_set_size = %6d, learning_rate = %lf\n",
//    sz0, sz1, sz2, sz3, TRAIN_SIZE, TEST_SIZE, learning_rate);
//  // 하이퍼 파라미터를 파일에 입력
//  fflush(fp_accuracy);
//
//  srand(time(NULL)); // random_state를 고정하고 하이퍼 파라미터와 함께 저장하도록 바꾸기
//
//  printf("Read Training Data\n");
//  Read_training_data();
//  printf("Init Parameters\n");
//  init_Parameters();
//  printf("Do Train Batch\n");
//  //Do_train(); // fit으로 정리
//  Do_train();
//
//  fclose(fp_accuracy);
//  return 0;
//}
//
//void Do_train() {
//  int index_minibatch = 0, train_set_index;
//  int start_of_this_minibatch, start_of_next_minibatch;
//  int epoch;
//
//  for (epoch = 0; epoch < MAX_EPOCH; epoch++) {
//    index_minibatch = 0;
//    do {
//      start_of_this_minibatch = index_minibatch * SIZE_MINIBATCH;
//      start_of_next_minibatch = start_of_this_minibatch + SIZE_MINIBATCH;
//      if (!(start_of_next_minibatch - 1 <= TRAIN_SIZE - 1)) {
//        break;
//      }
//      initialize_gradients_before_minibatch();
//      for (train_set_index = start_of_this_minibatch; train_set_index < start_of_next_minibatch; train_set_index++) {
//        if (train_set_index % 2000 == 1) {
//          printf("Train Supervised NN: At epoch %d. learning_rate = %f, Training example number = %d\n", epoch, learning_rate, train_set_index);
//        }
//        forward_compute(train_set_index);
//        backward_compute(train_set_index);
//        Accumulate_gradients_of_parameters(train_set_index);
//      }
//      update_parameters_for_minibatch();
//      index_minibatch++;
//    } while (1);
//
//    printf("\nSupervised NN system Testing after epoch= %d has started.\n", epoch);
//    double test_accuracy = Do_test();
//    fprintf(fp_accuracy, "After epoch= %d, learning_rate=%f, S-NN Accuracy = %f\n", epoch, learning_rate, test_accuracy);
//    fflush(fp_accuracy);
//    printf("S-NN Accuracy after epoch %3d = %f (stored in file Test_Accuracy.txt)\n\n", epoch, test_accuracy);
//  }
//}
//
//void forward_compute(int train_set_index) {
//  int i, j;
//  int row, column;
//  double denomi;
//
//  j = 0;
//  for (row = 0; row < ROW; row++) {
//    for (column = 0; column < COLUMN; column++) {
//      F0[j] = Train_Data[train_set_index][row][column] / 255;
//      j++;
//    }
//  }
//  // TODO : 일반화하여 재작성하기
//  for (i = 0; i < sz1; i++) {
//    F1[i] = Compute_sigmoid_F1_from_F0(i);
//    Sigmoid_deriv1[i] = F1[i] * (1 - F1[i]);
//  }
//  for (i = 0; i < sz2; i++) {
//    F2[i] = Compute_sigmoid_F2_from_F1(i);
//    Sigmoid_deriv2[i] = F2[i] * (1 - F2[i]);
//  }
//  for (i = 0; i < sz3; i++) {
//    F3[i] = Compute_IN3_from_F2(i);
//  }
//  denomi = 0.0;
//  for (i = 0; i < sz3; i++) {
//    denomi += exp(F3[i]);
//  }
//  for (i = 0; i < sz3; i++) {
//    SoftMax[i] = exp(F3[i]) / denomi;
//  }
//}
//
//void backward_compute(int train_set_index) {
//  int i, j, correct_label;
//  double temp;
//
//  // foward와 같이 계산?
//  for (i = 0; i < sz3; i++) {
//    Delta3[i] = SoftMax[i];
//  }
//
//  correct_label = Train_Label[train_set_index];
//  Delta3[correct_label]--;
//
//  for (i = 0; i < sz2; i++) {
//    temp = 0;
//    for (j = 0; j < sz3; j++) {
//      temp += Delta3[j] * W3[i][j];
//    }
//    Delta2[i] = Sigmoid_deriv2[i] * temp;
//  }
//
//  for (i = 0; i < sz1; i++) {
//    temp = 0;
//    for (j = 0; j < sz2; j++) {
//      temp += Delta2[j] * W2[i][j];
//    }
//    Delta1[i] = Sigmoid_deriv1[i] * temp;
//  }
//}
//
//void Accumulate_gradients_of_parameters(int train_set_index) {
//  int i, j;
//  for (i = 0; i < sz2; i++) {
//    for (j = 0; j < sz3; j++) {
//      W3_gradient[i][j] += Delta3[j] * F2[i];
//    }
//  }
//  for (i = 0; i < sz1; i++) {
//    for (j = 0; j < sz2; j++) {
//      W2_gradient[i][j] += Delta2[j] * F1[i];
//    }
//  }
//  for (i = 0; i < sz0; i++) {
//    for (j = 0; j < sz1; j++) {
//      W1_gradient[i][j] += Delta1[j] * F0[i];
//    }
//  }
//
//  for (i = 0; i < sz3; i++) {
//    B3_gradient[i] += Delta3[i];
//  }
//  for (i = 0; i < sz2; i++) {
//    B2_gradient[i] += Delta2[i];
//  }
//  for (i = 0; i < sz1; i++) {
//    B1_gradient[i] += Delta1[i];
//  }
//}
//
//void initialize_gradients_before_minibatch() {
//  int i, j;
//
//  for (i = 0; i < sz0; i++) {
//    for (j = 0; j < sz1; j++) {
//      W1_gradient[i][j] = 0;
//    }
//  }
//  for (i = 0; i < sz1; i++) {
//    for (j = 0; j < sz2; j++) {
//      W2_gradient[i][j] = 0;
//    }
//  }
//  for (i = 0; i < sz2; i++) {
//    for (j = 0; j < sz3; j++) {
//      W3_gradient[i][j] = 0;
//    }
//  }
//
//  for (i = 0; i < sz3; i++) {
//    B3_gradient[i] = 0;
//  }
//  for (i = 0; i < sz2; i++) {
//    B2_gradient[i] = 0;
//  }
//  for (i = 0; i < sz1; i++) {
//    B1_gradient[i] = 0;
//  }
//}
//
//void update_parameters_for_minibatch() {
//  int i, j;
//  // gradient로 계산하는 이유가 뭐임?
//
//  for (i = 0; i < sz2; i++) {
//    for (j = 0; j < sz3; j++) {
//      W3[i][j] -= learning_rate * W3_gradient[i][j];
//    }
//  }
//  for (i = 0; i < sz1; i++) {
//    for (j = 0; j < sz2; j++) {
//      W2[i][j] -= learning_rate * W2_gradient[i][j];
//    }
//  }
//  for (i = 0; i < sz0; i++) {
//    for (j = 0; j < sz1; j++) {
//      W1[i][j] -= learning_rate * W1_gradient[i][j];
//    }
//  }
//
//  for (i = 0; i < sz3; i++) {
//    B3[i] -= learning_rate * B3_gradient[i];
//  }
//  for (i = 0; i < sz2; i++) {
//    B2[i] -= learning_rate * B2_gradient[i];
//  }
//  for (i = 0; i < sz1; i++) {
//    B1[i] -= learning_rate * B1_gradient[i];
//  }
//}
//
//
//
//void init_Parameters() {
//  int i, j;
//  for (i = 0; i < sz0; i++) {
//    for (j = 0; j < sz1; j++) {
//      W1[i][j] = getRandNum();
//    }
//  }
//  for (i = 0; i < sz1; i++) {
//    for (j = 0; j < sz2; j++) {
//      W2[i][j] = getRandNum();
//    }
//  }
//  for (i = 0; i < sz2; i++) {
//    for (j = 0; j < sz3; j++) {
//      W3[i][j] = getRandNum();
//    }
//  }
//  for (i = 0; i < sz3; i++) {
//    B3[i] = getRandNum();
//  }
//  for (i = 0; i < sz2; i++) {
//    B2[i] = getRandNum();
//  }
//  for (i = 0; i < sz1; i++) {
//    B1[i] = getRandNum();
//  }
//}
//
//double getRandNum() {
//  double r = (double)rand();
//  int r2 = rand();
//  double sign;
//
//  if (r2 < RAND_MAX / 2) {
//    sign = -1.0;
//  }
//  else {
//    sign = 1.0;
//  }
//  return (r / (double)RAND_MAX) * sign;
//}
//
//double Compute_sigmoid_F1_from_F0(int i) {
//  double x = 0, res;
//  for (int j = 0; j < sz0; j++) {
//    x += W1[j][i] * F0[j];
//  }
//  x += B1[i];
//  IN1[i] = x;
//  res = 1.0 / (1 + exp(-x));
//  return res;
//}
//
//double Compute_sigmoid_F2_from_F1(int i) {
//  double x = 0, res;
//  for (int j = 0; j < sz1; j++) {
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
//  for (int j = 0; j < sz2; j++) {
//    x += W3[j][i] * F2[j];
//  }
//  x += B3[i];
//  IN3[i] = x;
//  return x;
//}
//
//double Do_test() {
//
//  double temp, max_value, accuracy;
//  int test_set_index, row, column, i;
//  int max_index, correct_count = 0;
//  int label_by_model;
//
//  for (test_set_index = 0; test_set_index < TEST_SIZE; test_set_index++) {
//
//    if (test_set_index % 1000 == 0) {
//      printf("Number of test examples tested = %d\n", test_set_index);
//    }
//    
//    for (row = 0; row < ROW; row++) {
//      for (column = 0; column < COLUMN; column++) {
//        F0[row * 28 + column] = Test_Data[test_set_index][row][column] / 255;
//      }
//    }
//    for (i = 0; i < sz1; i++) {
//      F1[i] = Compute_sigmoid_F1_from_F0(i);
//    }
//    for (i = 0; i < sz2; i++) {
//      F2[i] = Compute_sigmoid_F2_from_F1(i);
//    }
//    for (i = 0; i < sz3; i++) {
//      F3[i] = Compute_IN3_from_F2(i);
//    }
//
//    temp = 0.0;    
//    for (i = 0; i < sz3; i++) {
//      temp += exp(F3[i]);
//    }
//    for (i = 0; i < sz3; i++) {
//      SoftMax[i] = exp(F3[i]) / temp;
//    }
//    max_index = 0;
//    max_value = SoftMax[0];
//    for (i = 1; i < sz3; i++) {
//      if (max_value < SoftMax[i]) {
//        max_index = i;
//        max_value = SoftMax[i];
//      }
//    }
//    label_by_model = max_index;
//    if (label_by_model == Test_Label[test_set_index]) {
//      correct_count++;
//    }
//  }
//  accuracy = ((double)correct_count) / TEST_SIZE;
//  return accuracy;
//}
//
//void Read_training_data() {
//  FILE* fp;
//  int index, str_len;
//  int row, column, i, cp, temp;
//  char* ret_fgets;
//  char buff_line[1024];
//  char num_str[20];
//
//  fp = fopen("datasets/train.txt", "r");
//  if (!fp) {
//    printf("File Open Failed : train.txt\n");
//    getchar();
//    return;
//  }
//
//  for (index = 0; index < TRAIN_SIZE; index++) {
//    if (index % 10000 == 0) {
//      printf("Traing Index : %d\n", index);
//    }
//    ret_fgets = fgets(buff_line, 1024, fp);
//    if (!ret_fgets) {
//      printf("fets error1.\n");
//    }
//    str_len = strlen(buff_line);
//    if (str_len != 2) {
//      printf("wrong leng of error.\n");
//      getchar();
//    }
//    buff_line[--str_len] = '\0';
//    Train_Label[index] = atoi(buff_line);
//
//    for (row = 0; row < ROW; row++) {
//      ret_fgets = fgets(buff_line, 1024, fp);
//      if (!ret_fgets) {
//        printf("fgets error1.\n");
//      }
//      str_len = strlen(buff_line);
//      buff_line[--str_len] = '\0';
//
//      cp = 0;
//      for (column = 0; column < COLUMN; column++) {
//        while (buff_line[cp] == ' ') {
//          cp++;
//        }
//        i = 0;
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
//  fclose(fp);
//
//  fp = fopen("datasets/test.txt", "r");
//  if (!fp) {
//    printf("File open failed : test.txt\n");
//    return;
//  }
//
//  for (index = 0; index < TEST_SIZE; index++) {
//    if (index % 2000 == 0) {
//      printf("test data loading. data_index = %d\n", index);
//    }
//    fscanf(fp, "%d", &Test_Label[index]);
//
//    for (row = 0; row < ROW; row++) {
//      for (column = 0; column < COLUMN; column++) {
//        fscanf(fp, "%d", &temp);
//        Test_Data[index][row][column] = temp;
//      }
//    }
//  }
//  fclose(fp);
//}