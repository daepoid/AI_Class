#include<iostream>
#include<cstdlib>
#include<cmath>
#include<ctime>
using namespace std;
#pragma warning(disable: 4996)

/*
  배치 정규화 알고리즘 사용하기
  data와 label을 pair나 map으로 묶어서 사용하는 경우?
  learning rate, layer, random_state, epoch를 쉽게 조정할 수 있게 함수로 만들고 외부에서 처리할 수 있게 하기
  cross_validation, rmse 구현
  grid_search 구현 가능?
*/

#define train_set_size 60000
#define test_set_size 10000
#define ori_learning_rate 0.08
#define max_epoch 100
#define ROW 28
#define COLUMN 28

// training and test data
double learning_rate = ori_learning_rate;

int train_data[60000][28][28];
int train_label[60000];

int test_data[10000][28][28];
int test_label[10000];

// weight and bias parameters
const int sz0 = 784;
const int sz1 = 500;
const int sz2 = 400;
const int sz3 = 10;

double w1[sz0][sz1];
double w2[sz1][sz2];
double w3[sz2][sz3];

double b1[sz1];
double b2[sz2];
double b3[sz3];

// output of neurons
double f0[sz0];
double f1[sz1];
double f2[sz2];
double f3[sz3];
double softmax[sz3];

// TODO : 밑에 전역변수들 수정하기
// temp
double Sigmoid_deriv1[sz1];
double Sigmoid_deriv2[sz2];

// Weighted sums
double IN1[sz1];
double IN2[sz2];
double IN3[sz3];

// Delta variables
double Delta1[sz1];
double Delta2[sz2];
double Delta3[sz3];

FILE* fp_accuracy;

void Read_training_data();
void init_Parameters();
double getRandNum();
void Do_train();
double Compute_sigmoid_f1_from_f0(int i);
double Compute_sigmoid_f2_from_f1(int i);
double Compute_IN3_from_f2(int i);
double Do_test();

int main() {
  fp_accuracy = fopen("Test_Accuracy.txt", "a");
  if (!fp_accuracy) {
    printf("File Open Failed : Test_Accuracy.txt\n");
    return 0;
  }
  fprintf(fp_accuracy, "\n\nExperiment of building Supervised NN system has started.\n");
  fprintf(fp_accuracy, "sz0 = %5d, sz1 = %5d, sz2 = %5d, sz3 = %5d, train_set_size = %6d, test_set_size = %6d, learning_rate = %lf\n",
    sz0, sz1, sz2, sz3, train_set_size, test_set_size, learning_rate);
  // 하이퍼 파라미터를 파일에 입력

  fflush(fp_accuracy);
  srand(time(NULL)); // random_state를 고정하고 하이퍼 파라미터와 함께 저장하도록 바꾸기

  printf("Read Training Data\n");
  Read_training_data();
  printf("Init Parameters\n");
  init_Parameters();
  printf("DO Train\n");
  Do_train(); // fit으로 정리

  fclose(fp_accuracy);
  return 0;
}

void Do_train() {
  int i, j;
  int row, column;
  int epoch, train_set_index;
  int correct_label;
  double denomi, test_accuracy, temp;

  for (epoch = 0; epoch < max_epoch; epoch++) {
    for (train_set_index = 0; train_set_index < train_set_size; train_set_index++) {
      if (train_set_index % 2000 == 1) {
        printf("Train Supervised NN: At epoch %d. Ing_rate = %f, Training example number = %d\n", epoch, learning_rate, train_set_index);
      }
      j = 0;
      for (row = 0; row < ROW; row++) {
        for (column = 0; column < COLUMN; column++) {
          f0[j] = train_data[train_set_index][row][column] / 255;
          j++;
        }
      }
      // TODO : 일반화하여 재작성하기
      for (i = 0; i < sz1; i++) {
        f1[i] = Compute_sigmoid_f1_from_f0(i);
        Sigmoid_deriv1[i] = f1[i] * (1 - f1[i]);
      }
      for (i = 0; i < sz2; i++) {
        f2[i] = Compute_sigmoid_f2_from_f1(i);
        Sigmoid_deriv2[i] = f2[i] * (1 - f2[i]);
      }
      for (i = 0; i < sz3; i++) {
        f3[i] = Compute_IN3_from_f2(i);
      }
      denomi = 0.0;
      for (i = 0; i < sz3; i++) {
        denomi += exp(f3[i]);
      }
      for (i = 0; i < sz3; i++) {
        Delta3[i] = exp(f3[i]) / denomi;
      }

      correct_label = train_label[train_set_index];
      Delta3[correct_label]--;

      for (i = 0; i < sz2; i++) {
        temp = 0;
        for (j = 0; j < sz3; j++) {
          temp += Delta3[j] * w3[i][j];
        }
        Delta2[i] = Sigmoid_deriv2[i] * temp;
      }

      for (i = 0; i < sz1; i++) {
        temp = 0;
        for (j = 0; j < sz2; j++) {
          temp += Delta2[j] * w2[i][j];
        }
        Delta1[i] = Sigmoid_deriv1[i] * temp;
      }
      // TODO : for문을 한번에 처리하기
      for (i = 0; i < sz2; i++) {
        for (j = 0; j < sz3; j++) {
          // TODO : learning_rate 뒤에 double로 잘 계산되는지 확인
          temp = learning_rate * (Delta3[j] * f2[i]);
          w3[i][j] -= temp;
        }
      }
      for (i = 0; i < sz1; i++) {
        for (j = 0; j < sz2; j++) {
          temp = learning_rate * (Delta2[j] * f1[i]);
          w2[i][j] -= temp;
        }
      }
      for (i = 0; i < sz0; i++) {
        for (j = 0; j < sz1; j++) {
          temp = learning_rate * (Delta1[j] * f0[i]);
          w1[i][j] -= temp;
        }
      }
      for (i = 0; i < sz3; i++) {
        b3[i] -= learning_rate * Delta3[i];
      }
      for (i = 0; i < sz2; i++) {
        b2[i] -= learning_rate * Delta2[i];
      }
      for (i = 0; i < sz1; i++) {
        b1[i] -= learning_rate * Delta1[i];
      }
    }
    printf("\nSupervised NN system Testing after epoch= %d has started.\n", epoch);
    test_accuracy = Do_test();
    fprintf(fp_accuracy, "After epoch= %d, Ing_rate=%f, S-NN Accuracy = %f\n", epoch, learning_rate, test_accuracy);
    fflush(fp_accuracy);
    printf("S-NN Accuracy after epoch %3d = %f (stored in file Test_Accuracy.txt)\n\n", epoch, test_accuracy);
  }
}

void init_Parameters() {
  int i, j;
  for (i = 0; i < sz0; i++) {
    for (j = 0; j < sz1; j++) {
      w1[i][j] = getRandNum();
    }
  }
  for (i = 0; i < sz1; i++) {
    for (j = 0; j < sz2; j++) {
      w2[i][j] = getRandNum();
    }
  }
  for (i = 0; i < sz2; i++) {
    for (j = 0; j < sz3; j++) {
      w3[i][j] = getRandNum();
    }
  }
  for (i = 0; i < sz3; i++) {
    b3[i] = getRandNum();
  }
  for (i = 0; i < sz2; i++) {
    b2[i] = getRandNum();
  }
  for (i = 0; i < sz1; i++) {
    b1[i] = getRandNum();
  }
}

double getRandNum() {
  double r = (double)rand();
  int r2 = rand();
  double sign;

  if (r2 < RAND_MAX / 2) {
    sign = -1.0;
  }
  else {
    sign = 1.0;
  }
  return (r / (double)RAND_MAX) * sign;
}

double Compute_sigmoid_f1_from_f0(int i) {
  double x = 0, res;
  for (int j = 0; j < sz0; j++) {
    x += w1[j][i] * f0[j];
  }
  x += b1[i];
  IN1[i] = x;
  res = 1.0 / (1 + exp(-x));
  return res;
}

double Compute_sigmoid_f2_from_f1(int i) {
  double x = 0, res;
  for (int j = 0; j < sz1; j++) {
    x += w2[j][i] * f1[j];
  }
  x += b2[i];
  IN2[i] = x;
  res = 1.0 / (1 + exp(-x));
  return res;
}

double Compute_IN3_from_f2(int i) {
  double x = 0;
  for (int j = 0; j < sz2; j++) {
    x += w3[j][i] * f2[j];
  }
  x += b3[i];
  IN3[i] = x;
  return x;
}

double Do_test() {

  double temp, max_value, accuracy;
  int test_set_index, row, column, i;
  int max_index, correct_count = 0;
  int label_by_model;

  for (test_set_index = 0; test_set_index < test_set_size; test_set_index++) {

    if (test_set_index % 1000 == 0) {
      printf("Number of test examples tested = %d\n", test_set_index);
    }
    
    for (row = 0; row < ROW; row++) {
      for (column = 0; column < COLUMN; column++) {
        f0[row * 28 + column] = test_data[test_set_index][row][column] / 255;
      }
    }
    for (i = 0; i < sz1; i++) {
      f1[i] = Compute_sigmoid_f1_from_f0(i);
    }
    for (i = 0; i < sz2; i++) {
      f2[i] = Compute_sigmoid_f2_from_f1(i);
    }
    for (i = 0; i < sz3; i++) {
      f3[i] = Compute_IN3_from_f2(i);
    }

    temp = 0.0;    
    for (i = 0; i < sz3; i++) {
      temp += exp(f3[i]);
    }
    for (i = 0; i < sz3; i++) {
      softmax[i] = exp(f3[i]) / temp;
    }
    max_index = 0;
    max_value = softmax[0];
    for (i = 1; i < sz3; i++) {
      if (max_value < softmax[i]) {
        max_index = i;
        max_value = softmax[i];
      }
    }
    label_by_model = max_index;
    if (label_by_model == test_label[test_set_index]) {
      correct_count++;
    }
  }
  accuracy = ((double)correct_count) / test_set_size;
  return accuracy;
}

void Read_training_data() {
  FILE* fp;
  int index, str_len;
  int row, column, i, cp, temp;
  char* ret_fgets;
  char buff_line[1024];
  char num_str[20];

  fp = fopen("datasets/train.txt", "r");
  if (!fp) {
    printf("File Open Failed : train.txt\n");
    getchar();
    return;
  }

  for (index = 0; index < train_set_size; index++) {
    if (index % 10000 == 0) {
      printf("Traing Index : %d\n", index);
    }
    ret_fgets = fgets(buff_line, 1024, fp);
    if (!ret_fgets) {
      printf("fets error1.\n");
    }
    str_len = strlen(buff_line);
    if (str_len != 2) {
      printf("wrong leng of error.\n");
      getchar();
    }
    buff_line[--str_len] = '\0';
    train_label[index] = atoi(buff_line);

    for (row = 0; row < ROW; row++) {
      ret_fgets = fgets(buff_line, 1024, fp);
      if (!ret_fgets) {
        printf("fgets error1.\n");
      }
      str_len = strlen(buff_line);
      buff_line[--str_len] = '\0';

      cp = 0;
      for (column = 0; column < COLUMN; column++) {
        while (buff_line[cp] == ' ') {
          cp++;
        }
        i = 0;
        while (isdigit(buff_line[cp])) {
          num_str[i] = buff_line[cp];
          i++;
          cp++;
        }
        num_str[i] = '\0';
        train_data[index][row][column] = atof(num_str);
      }
    }
  }
  fclose(fp);

  fp = fopen("datasets/test.txt", "r");
  if (!fp) {
    printf("File open failed : test.txt\n");
    return;
  }

  for (index = 0; index < test_set_size; index++) {
    if (index % 2000 == 0) {
      printf("test data loading. data_index = %d\n", index);
    }
    fscanf(fp, "%d", &test_label[index]);

    for (row = 0; row < ROW; row++) {
      for (column = 0; column < COLUMN; column++) {
        fscanf(fp, "%d", &temp);
        test_data[index][row][column] = temp;
      }
    }
  }
  fclose(fp);
}