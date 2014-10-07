(ns anna.core
  (:use [clojure.core.matrix]
        [clojure.pprint]
        [clojure.java.io]
        [clojure.java.shell :only [sh]])
  (:require [clojure.core.matrix.operators :as Mtrx]
            [clojure.string :as str]
            [taoensso.timbre :as log]
            [clojure.tools.nrepl.server :only [start-server stop-server] :as nrepl]
            [lamina.core :as lamina]
            [aleph.http :as aleph]
            [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))


(def ^:dynamic *talk-to-me* true)
(def ^:dynamic *log-progress* true)
(def ^:dynamic *username* (System/getProperty "user.name"))
(def ^:dynamic *osname* (System/getProperty "os.name"))
(def ^:dynamic *log-output* (clojure.java.io/writer "/tmp/anna.log"))
(comment (.close *log-output*))

(defn- say
  "Text-to-Speech output if you run this on Mac OS X"
  ([msg]
     (say msg false))
  ([msg blocking]
      (when (and *talk-to-me* (= "Mac OS X" *osname*))
        (let [say-it (fn [s] (sh "say" "-v" "Samantha" s))]
          (if (= true blocking)
            (future (say-it msg))
            (say-it msg))))))

(defn- log-msg
  [& msg]
  (when (= true *log-progress*)
    (binding [*out* *log-output*]
      (log/info msg))))

(defn- mapval
  "Map input interval to output interval"
  [in-min in-max out-min out-max x]
  (+ (/ (* (- x in-min) (- out-max out-min))
        (- in-max in-min))
     out-min))

(defn- num-to-vector
  "Creates a vector of size, sets all elements to 0.0, 
except the xth element, which is set to 1.0
Example: size=3 x=2 returns [[0.0] [0.0] [1.0]]."
  [size x]
  (assoc (into [] (map #(vector %)
                       (repeat size 0.0)))
    x
    [1.0]))

(defn- load-xor-data
"Test data for XOR network"
  []
  [{:input [[0] [0]] :output [[0]]}
   {:input [[1] [0]] :output [[1]]}
   {:input [[0] [1]] :output [[1]]}
   {:input [[1] [1]] :output [[0]]}])

(defn- transform-mnist-data
  [line]
  (let [scale-0-to-1 (partial mapval 0 255 0.0 1.0)
        [output & input] (str/split line #",")
        input-vector (into [] (map #(vector (scale-0-to-1 (Integer/parseInt %)))
                                       input))
        output-vector (num-to-vector 10 (Integer/parseInt output))]
    {:input input-vector :output output-vector}))

(defn- load-data
  [filename]
  (let [num-lines (atom 0)
        dataset (atom [])]
    (with-open [rdr (reader filename)]
      (doseq [line (line-seq rdr)]
        (swap! num-lines inc)
        (swap! dataset conj line)
        (when (= 0 (mod @num-lines 2500))
          (log-msg @num-lines))))
    @dataset))

(defn save-neuralnet
  "Serialize neural network to file"
  [filename form]
  (let [f (file filename)]
    (spit f (pr-str (update-in form
                               [:options :transformer-fn]
                               (constantly nil))))))

(defn load-neuralnet
  "Load neural network from file"
  [filename]
  (let [f (file filename)]
    (read-string (slurp f))))


(defn- g
  "Sigmoid function"
  [z]
  (emap #(/ 1 (+ 1 (Math/exp (- %)))) z)) ;TODO: Better vectorized version

(defn- g'
  "Sigmoid derivative function"
  [z]
  (Mtrx/* (g z) (Mtrx/- 1 (g z))))

(defn- zeros
  "Create a matrix with dimension n x m and fill with 0.0"
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] 0.0)))

(defn- ones
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] 1.0)))

(defn- rand-epsilon
  [epsilon]
  (- (rand (* 2 epsilon)) epsilon))

(defn- randos
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] (rand-epsilon 1))))

(defn- matrix-mult
  [a b]
  ;(log-msg "a" (shape a) (matrix? a)) (pm a)
  ;(log-msg "b" (shape b) (matrix? b)) (pm b)
  (let [result (mmul a b)]
    (if (matrix? result)
      result
      (matrix [result]))))

(defn- bind-bias
  "Add a bias node to an activations vector"
  [activations]
  (into (matrix [[1]]) activations))

(defn- mse
  [errors]
  (/ (reduce + 0 (flatten (pow errors 2))) (count errors)))

(defrecord Options [learning-rate
                    max-iterations
                    error-threshold
                    momentum
                    mini-batch-size
                    transformer-fn
                    num-validation-records])

(defn make-options
  ([]
     (make-options {}))
  ([{:keys [learning-rate
            max-iterations
            error-threshold
            momentum
            mini-batch-size
            transformer-fn
            num-validation-records]
     :or {learning-rate 0.7
          max-iterations 2000
          error-threshold 0.001
          momentum 0.3
          mini-batch-size 4
          transformer-fn nil
          num-validation-records 0}
     :as params}] 
     (Options. learning-rate max-iterations error-threshold
               momentum mini-batch-size transformer-fn num-validation-records)))

(defprotocol Forwardpropagation
  (calc-activations [this input]))

(defprotocol Backpropagation
  (calc-output-delta [this ideal-output])
  (calc-hidden-delta [this weights delta])
  (calc-gradients [this activations])
  (calc-new-weights [this learning-rate momentum]))

(defprotocol ANN
  (forwardpropagation [this input])
  (backpropagation [this ideal-output])
  (calc-error [this ideal-output])
  (update-gradients [this input])
  (update-weights [this])
  (validate [this validation-data])
  (train [this training-data])
  (output-layer [this])
  (global-error [this ideal-output])
  (exec [this input]))

(defrecord Layer [weights weighted-sum activations delta gradients]
  Forwardpropagation
  (calc-activations
    [this input]
    (let [weighted-sum (matrix-mult weights (bind-bias input))
          new-activations (g weighted-sum)]
      (assoc this :weighted-sum weighted-sum :activations new-activations)))
  Backpropagation
  (calc-output-delta
    [this ideal-output]
    (let [output-error (Mtrx/- (Mtrx/- (:activations this) ideal-output))
          output-delta (Mtrx/* output-error (g' (:weighted-sum this)))]
      (assoc this :delta output-delta :error output-error)))
  (calc-hidden-delta
    [this weights delta]
    (let [hidden-delta (Mtrx/* (matrix-mult (transpose weights)
                                            delta)
                               (g' (bind-bias (:weighted-sum this))))] ;TODO: Bias?
      ;call (rest hidden-delta) to remove the bias from the result
      (assoc this :delta (rest hidden-delta))))
  (calc-gradients
    [this activations]
    (let [gradients (matrix (matrix-mult (:delta this)
                                         (transpose
                                          (bind-bias activations))))]
      (assoc this :gradients gradients)))
  (calc-new-weights
    [this learning-rate momentum]
    (let [new-weights (Mtrx/+ (:weights this)
                              (Mtrx/* learning-rate
                                      (:gradients this)))]
      (assoc this :weights new-weights)))) 

(defn- rounded-results
  [output]
  (map #(Math/round %) (map first output)))

(defrecord NeuralNetwork [layers iterations error options]
  ANN
  (forwardpropagation
    [this input]
    (loop [layers (:layers this)
           accu []
           activations input]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              remaining-layers (rest layers)
              updated-layer (calc-activations cur-layer activations)
              new-activations (:activations updated-layer)]
          (recur remaining-layers
                 (conj accu updated-layer)
                 new-activations)))))
  (backpropagation
    [this ideal-output]
    (let [reversed-layers (reverse (:layers this))
          output-layer (first reversed-layers)
          new-output-layer (calc-output-delta output-layer ideal-output)]
      (loop [layers' (rest reversed-layers)
             accu [new-output-layer]
             prev-layer new-output-layer]
        (if (empty? layers')
          (assoc this :layers (reverse accu))
          (let [cur-layer (first layers')
                remaining-layers (rest layers')
                updated-layer (calc-hidden-delta cur-layer
                                                 (:weights prev-layer)
                                                 (:delta prev-layer))]
            (recur remaining-layers
                   (conj accu updated-layer)
                   updated-layer))))))
  (calc-error
    [this ideal-output]
    (mse (global-error this ideal-output)))
  (update-gradients
    [this input]
    (loop [layers (:layers this)
           prev-activations input
           accu []]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              updated-gradients (calc-gradients cur-layer
                                                prev-activations)]
          (recur (rest layers)
                 (:activations cur-layer)
                 (conj accu updated-gradients))))))
  (update-weights
    [this]
    (loop [layers (:layers this)
           accu []]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              updated-layer (calc-new-weights cur-layer
                                              (:learning-rate (:options this))
                                              (:momentum (:options this)))]
          (recur (rest layers)
                 (conj accu updated-layer))))))
  (validate
    [this validation-data]
    (doseq [record validation-data]
      (forwardpropagation this record)))
  (train
    [this training-data]
    (let [opt (:options this)
          mini-batch-size (:mini-batch-size opt)
          max-iterations (:max-iterations opt)
          error-threshold (:error-threshold opt)
          transformer-fn (if-let [f (:transformer-fn opt)] 
                           (fn [x] (f x))
                           (fn [x] x))
          num-validation-records (:num-validation-records opt)
          datasets (split-at num-validation-records
                             training-data)
          validation-dataset (first datasets)
          training-dataset (second datasets)]
      (say (str "Training with " (count training-dataset) " records."))
      (say (str "Validating with " (count validation-dataset) " records."))
      (loop [epoch 0
             outer-nn this]
        (log-msg "Epoch:" epoch "Error:" (:error outer-nn))
        (if (or (< (:error outer-nn) error-threshold)
                (> epoch max-iterations))
          outer-nn
          (recur
           (inc epoch)
           (loop [mini-batches (partition mini-batch-size
                                          (shuffle training-dataset))
                  cur-mini-batch (first mini-batches)
                  num-batch 0
                  inner-nn outer-nn]
             (log-msg (str "Epoch: " epoch ", Mini-Batch: " num-batch ", Error: " (:error inner-nn)))
;             (say (str "Mini batch number " num-batch " started."))
             (if (empty? mini-batches)
               (do
                 (validate inner-nn validation-dataset)
                 inner-nn)
               (recur
                (rest mini-batches)
                (first (rest mini-batches))
                (inc num-batch)
                (loop [sample (transformer-fn (first cur-mini-batch))
                       remaining-samples (rest cur-mini-batch)
                       total-error 0
                       inner-nn' inner-nn]
                    (if (empty? remaining-samples)
                      inner-nn'
                      (let [nn1 (forwardpropagation inner-nn' (:input sample))
                            nn2 (backpropagation nn1 (:output sample))
                            nn3 (update-gradients nn2 (:input sample))
                            avg-error (/ (/ (calc-error nn3 (:output sample))
                                            mini-batch-size) 2)
                            summed-avg-error (+ total-error avg-error)
                            nn4 (assoc nn3 :error summed-avg-error)]
                        ;; (log-msg sample "=" (output nn4)
                        ;;          "Error" (global-error nn4 (:output sample))
                        ;;          "Error:" summed-avg-error)
                        (recur (transformer-fn (first remaining-samples))
                               (rest remaining-samples)
                               summed-avg-error
                               (update-weights nn4)))))))))))))
  (output-layer
    [this]
    (-> this :layers last :activations))
  (global-error
    [this ideal-output]
    (Mtrx/- ideal-output (output-layer this)))
  (exec
    [this input]
    (let [result (-> (forwardpropagation this input)
                     output-layer)]
      (map #(Math/round %) (map first result)))))

(defn make-neuralnet
  ([nodes-in-layer]
     (make-neuralnet nodes-in-layer (make-options)))
  ([nodes-in-layer options]
      {:pre [(and
              (vector? nodes-in-layer)
              (not (empty? nodes-in-layer))
              (> (count nodes-in-layer) 1))]}
      (let [synapses (partition 2 1 nodes-in-layer)
            layers (map (fn [[cols rows]]
                          (let [col-with-bias (inc cols)]
                            ;add column 0 for the bias weights
                            (Layer. (matrix (randos rows col-with-bias)) 
                                    (matrix (zeros rows 1))
                                    (matrix (zeros rows 1))
                                    (matrix (zeros rows col-with-bias))
                                    (matrix (randos rows col-with-bias)))))
                        synapses)]
        (do
          (say "Neural Network created.")
          (NeuralNetwork. layers 0 1.0 options)))))

(defmacro defneuralnet
  [name layers & options]
  (let [opts (first options)]
    (if (nil? opts)
      `(def ~name (make-neuralnet ~layers))
      `(def ~name (make-neuralnet ~layers ~opts)))))

;; (defn- perceptron-test
;;   []
;;   (let [training [{:bias 1 :input [1 1] :output 1}
;;                   {:bias 1 :input [0 0] :output 0}
;;                   {:bias 1 :input [1 0] :output 0}
;;                   {:bias 1 :input [0 1] :output 0}]
;;         learning-rate 0.3
;;         acivate (fn [n] (if (<= n 0)
;;                           0
;;                           1))
;;         weights [0.23456 0.78901 0.56789]
;;         feed-forward (fn [input]
;;                        (reduce + (map (fn [i w] (* i w))
;;                                       input weights)))
;;         train (fn []
;;                 (map #(feed-forward (:input %))
;;                      training-data))]))

(defn- xor-test
  []
  (binding [*talk-to-me* false]
    (let [nn0 (make-neuralnet [2 3 1])
          nn1 (train nn0 (load-xor-data))
          check (fn [result expected] (= expected
                                         (Math/round (first (first result)))))]
      (binding [*talk-to-me* true]
        (let [result (list (check (exec nn1 [[0] [0]]) 0)
                           (check (exec nn1 [[1] [0]]) 1)
                           (check (exec nn1 [[0] [1]]) 1)
                           (check (exec nn1 [[1] [1]]) 0))]
          (if (every? true? result)
            (say "Ex OR test successful")
            (say "Ex OR was not test successful")))))))

(defn- mnist-test
  []
  (binding [*talk-to-me* true]
    (let [test-data (load-data "/Users/stefan/proj/anna/mnist_train.csv")
          nn0 (make-neuralnet [784 15 10] (make-options {:mini-batch-size 10
                                                         :learning-rate 3.0
                                                         :max-iterations 30
                                                         :transformer-fn transform-mnist-data
                                                         #_:num-validation-records #_1000}))
          nn1 (train nn0 test-data)
          check (fn [result expected] (= expected
                                         (Math/round (first (first result)))))]
      nn1)))

(defn hello-world [channel request]
  (lamina/enqueue channel
    {:status 200
     :headers {"content-type" "text/html"}
     :body "Hello World!"}))

;(start-http-server hello-world {:port 8008})

;(defonce nrepl-server (start-server :port 7888))

(def cli-options
  ;; An option with a required argument
  [["-p" "--port PORT" "Port number"
    :default 80
    :parse-fn #(Integer/parseInt %)
    :validate [#(< 0 % 0x10000) "Must be a number between 0 and 65536"]]
   ;; A non-idempotent option
   ["-v" nil "Verbosity level"
    :id :verbosity
    :default 0
    :assoc-fn (fn [m k _] (update-in m [k] inc))]
   ;; A boolean option defaulting to nil
   ["-h" "--help"]])

(defn -main
  [& args]
  (pprint (parse-opts args cli-options))
  (nrepl/start-server :port 7888))
