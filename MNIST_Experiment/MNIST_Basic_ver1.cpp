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
//
//*/
//
//#define TRAIN_SIZE 60000
//#define TEST_SIZE 10000
//
//#define MAX_EPOCH 10
//#define ROW 28
//#define COLUMN 28
//
//// training and test data
//double learning_rate = 0.08;
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
//
//  fflush(fp_accuracy);
//  srand(42); // random_state를 고정하고 하이퍼 파라미터와 함께 저장하도록 바꾸기
//
//  printf("Read Training Data\n");
//  Read_training_data();
//  printf("Init Parameters\n");
//  init_Parameters();
//  printf("DO Train\n");
//  Do_train(); // fit으로 정리
//
//  fclose(fp_accuracy);
//  return 0;
//}
//
//void Read_training_data() {
//  FILE* fp;
//  int data_index, str_len;
//  int row, column, i, temp;
//  char* ret_fgets;
//  char buff_line[1024];
//  char num_str[20];
//
//  fp = fopen("datasets/train.txt", "r");
//  if (!fp) {
//    printf("File Open Failed : train.txt\n");
//    return ;
//  }
//
//  for (data_index = 0; data_index < TRAIN_SIZE; data_index++) {
//    if (data_index % 5000 == 0) {
//      printf("Traing Index : %d\n", data_index);
//    }
//    ret_fgets = fgets(buff_line, 1024, fp);
//    if (!ret_fgets) {
//      printf("fets error1\n");
//    }
//    str_len = strlen(buff_line);
//    if (str_len != 2) {
//      printf("str_len error\n");
//    }
//    buff_line[--str_len] = '\0';
//    Train_Label[data_index] = atoi(buff_line);
//    
//    for (row = 0; row < ROW; row++) {
//      ret_fgets = fgets(buff_line, 1024, fp);
//      if (!ret_fgets) {
//        printf("fgets error1.\n");
//      }
//      str_len = strlen(buff_line);
//      buff_line[--str_len] = '\0';
//      
//      int cp = 0;
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
//        Train_Data[data_index][row][column] = atof(num_str);
//      }
//    }
//  }
//  fclose(fp);
//
//  fp = fopen("datasets/test.txt", "r");
//  if (!fp) {
//    printf("File open failed : test.txt\n");
//    return ;
//  }
//
//  for (data_index = 0; data_index < TEST_SIZE; data_index++) {
//    if (data_index % 2000 == 0) {
//      printf("test data loading. data_index = %d\n", data_index);
//    }
//    fscanf(fp, "%d", &Test_Label[data_index]);
//
//    for (row = 0; row < ROW; row++) {
//      for (column = 0; column < COLUMN; column++) {
//        fscanf(fp, "%d", &temp);
//        Test_Data[data_index][row][column] = temp;
//      }
//    }
//  }
//  fclose(fp);
//}
//
//void init_Parameters() {
//  for (int i = 0; i < sz0; i++) {
//    for (int j = 0; j < sz1; j++) {
//      W1[i][j] = getRandNum();
//      if (i < sz1 && j < sz2) {
//        W2[i][j] = getRandNum();
//      }
//      if (i < sz2 && j < sz3) {
//        W3[i][j] = getRandNum();
//      }
//    }
//  }
//  // 따로 만드는게 나은가 이니면 한 번에 하는게 나은가?
//  for (int i = 0; i < sz1; i++) {
//    B1[i] = getRandNum();
//    if (i < sz2) {
//      B2[i] = getRandNum();
//    }
//    if (i < sz3) {
//      B3[i] = getRandNum();
//    }    
//  }
//}
//
//double getRandNum() {  
//  return (rand() < (RAND_MAX / 2) ? -0.5 : 0.5) * (rand() / (double)RAND_MAX);
//}
//
//void Do_train() {
//  int row, column;
//
//  for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
//    for (int train_set_index = 0; train_set_index < TRAIN_SIZE; train_set_index++) {
//      if (train_set_index % 20000 == 1) {
//
//      }
//      for (row = 0; row < ROW; row++) {
//        for (column = 0; column < COLUMN; column++) {
//          F0[row * ROW + column] = (Train_Data[train_set_index][row][column] / 255.0 - 0.5) * 2;
//        }
//      }
//
//      // TODO : 일반화하여 재작성하기
//      int i, j;
//      for (i = 0; i < sz1; i++) {
//        F1[i] = Compute_sigmoid_F1_from_F0(i);
//        Sigmoid_deriv1[i] = F1[i] * (1 - F1[i]);
//      }
//      for (i = 0; i < sz2; i++) {
//        F2[i] = Compute_sigmoid_F2_from_F1(i);
//        Sigmoid_deriv2[i] = F2[i] * (1 - F2[i]);
//      }
//      for (i = 0; i < sz3; i++) {
//        F3[i] = Compute_IN3_from_F2(i);
//      }
//
//      double denomi = 0.0;
//      for (i = 0; i < sz3; i++) {
//        denomi += exp(F3[i]);
//      }
//      for (i = 0; i < sz3; i++) {
//        Delta3[i] = exp(F3[i]) / denomi;
//      }
//
//      Delta3[Train_Label[train_set_index]]--;
//      double temp;
//      for (i = 0; i < sz2; i++) {
//        temp = 0;
//        for (j = 0; j < sz3; j++) {
//          temp += Delta3[j] * W3[i][j];
//        }
//        Delta2[i] = Sigmoid_deriv2[i] * temp;
//      }
//
//      for (i = 0; i < sz1; i++) {
//        temp = 0;
//        for (j = 0; j < sz2; j++) {
//          temp += Delta2[j] * W2[i][j];
//        }
//        Delta1[i] = Sigmoid_deriv1[i] * temp;
//      }
//
//      // TODO : for문을 한번에 처리하기
//      for (i = 0; i < sz2; i++) {
//        for (j = 0; j < sz3; j++) {
//          // TODO : learning_rate 뒤에 double로 잘 계산되는지 확인
//          W3[i][j] -= learning_rate * (Delta3[j] * F2[i]);
//        }
//      }
//      for (i = 0; i < sz1; i++) {
//        for (j = 0; j < sz2; j++) {
//          W2[i][j] -= learning_rate * (Delta2[j] * F1[i]);
//        }
//      }
//      for (i = 0; i < sz0; i++) {
//        for (j = 0; j < sz1; j++) {
//          W1[i][j] -= learning_rate * (Delta1[j] * F0[i]);
//        }
//      }
//      
//      for (int i = 0; i < sz3; i++) {
//        B3[i] -= learning_rate * Delta3[i];
//      }
//      for (int i = 0; i < sz2; i++) {
//        B2[i] -= learning_rate * Delta2[i];
//      }
//      for (int i = 0; i < sz1; i++) {
//        B1[i] -= learning_rate * Delta1[i];
//      }
//    }
//    printf("\nSupervised NN system Testing after epoch= %d has started.\n", epoch);
//    double test_accuracy = Do_test();
//    fprintf(fp_accuracy, "After epoch= %d, Ing_rate=%f, S-NN Accuracy = %f\n", epoch, learning_rate, test_accuracy);
//    fflush(fp_accuracy);
//    printf("S-NN Accuracy after epoch %3d = %f (stored in file Test_Accuracy.txt\n\n", epoch, test_accuracy);
//  }
//}
//
//double Compute_sigmoid_F1_from_F0(int i) {
//  double x = 0;
//  for (int j = 0; j < sz0; j++) {
//    x += W1[j][i] * F0[j];
//  }
//  x += B1[i];
//  IN1[i] = x;
//  return 1.0 / (double)(1 + exp(-x));
//}
//
//double Compute_sigmoid_F2_from_F1(int i) {
//  double x = 0;
//  for (int j = 0; j < sz1; j++) {
//    x += W2[j][i] * F1[j];
//  }
//  x += B2[i];
//  IN2[i] = x;
//  return 1.0 / (double)(1 + exp(-x));
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
//  double temp = 0.0, max_value;
//  int data_index, row, column, i;
//  int max_index, correct_count = 0;
//
//  for (data_index = 0; data_index < TEST_SIZE; data_index++) {
//
//    if (data_index % 1000 == 0) {
//      printf("Number of test examples tested = %d\n", data_index);
//    }
//    
//    for (row = 0; row < ROW; row++) {
//      for (column = 0; column < COLUMN; column++) {
//        F0[row * 28 + column] = (Test_Data[data_index][row][column] / 255 - 0.5) * 2;
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
//    if (Test_Label[data_index] == max_index) {
//      correct_count++;
//    }
//  }
//  return (double)correct_count / TEST_SIZE;
//}