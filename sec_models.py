from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, precision_score, confusion_matrix, accuracy_score

#custom class to select a model
class ModelSelector:
  def __init__(self, model_name):
    self.model_name = model_name
    self.model = None
    self.getModel()

    self.X_test = None # this is necessary for ANN model only. Otherwise we dont need it.

    self.accuracy = None
    self.predictions = None
    self.f1_score = None
    self.precision = None
    self.conf_matrix = None

  def getModel(self):
    if self.model_name == 'LR': #check
        self.model = LogisticRegression()
    elif self.model_name == 'SVM': #check
        self.model = SVC()
    elif self.model_name == "RFC": #check
      self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif self.model_name == "DTC": #check
      self.model = DecisionTreeClassifier(random_state=42)
    elif self.model_name == 'KNN': #optional
        self.model = KNeighborsClassifier(n_neighbors=5)
    elif self.model_name == 'NB': #optional
        self.model = GaussianNB()
    else:
        raise ValueError(f'Model {self.model_name} is not recognized.')

  def fit(self, X_train, y_train):
    self.model.fit(X_train, y_train)

  def predict(self, X_test):
    self.X_test = X_test
    if self.model_name == 'ANN':
        self.predictions = (self.model.predict(X_test) > 0.5).astype(int)
    else:
        self.predictions = self.model.predict(X_test)

  def evaluate(self, y_test):
    #calculate accuracy
    self.accuracy = accuracy_score(self.predictions, y_test)

    #calculate f1-score
    self.f1_score = f1_score(y_test, self.predictions, average='binary')

    # Calculate Precision
    self.precision = precision_score(y_test, self.predictions, average='binary')

    # Compute Confusion Matrix
    self.conf_matrix = confusion_matrix(y_test, self.predictions)