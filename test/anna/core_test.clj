(ns anna.core-test
  (:require [clojure.test :refer :all]
            [anna.core :refer :all])
  ;(:require [clojure.math.numeric-tower :as math])
  ;(:use midje.sweet)
  (:use [clojure.core.matrix]
        [clojure.pprint])
  (:require [clojure.core.matrix.operators :as Mtrx]))


(def training-data [{:input [[0] [0]] :output [[0]]}
                    {:input [[1] [0]] :output [[1]]}
                    {:input [[0] [1]] :output [[1]]}
                    {:input [[1] [1]] :output [[0]]}])

(deftest single-step
  (testing "Single-step of one iteration")
  (let [input (matrix [[1] [1]])
        output (matrix [[1]])
        nn0 (make-neuralnetwork [2 3 1])
        nn1 (forwardpropagation nn0 input)
        nn2 (backpropagation nn1 output)
        nn3 (update-gradients nn2 input)
        nn4 (update-weights nn3)
        first-weights (fn [nn] (-> nn :layers first :weights))
        first-column (fn [nn] (get-column (-> nn :layers first :weights) 0))]
    (is (= 0 (compare (first-column nn2) (first-column nn3))))))

(deftest precalculated-XOR
  (testing "Precalculated XOR"
    (let [nn (make-neuralnetwork [2 2 1])
          input (matrix [[1.0]
                         [0.0]])
          output (matrix [[1.0]])
          w1 (matrix [[-0.46 -0.07 0.22]
                      [0.10 0.94 0.46]])
          z1 (matrix [[-0.53]
                      [1.05]])
          a1 (matrix [[0.37]
                      [0.74]])
          d1 (matrix [[0.0]
                      [0.0]])
          g1 (matrix [[0.0 0.0 0.0]
                      [0.0 0.0 0.0]])
          w2 (matrix [[0.78 -0.22 0.58]])
          z2 (matrix [[1.1254]])
          a2 (matrix [[0.75]])
          d2 (matrix [[0.0]])
          g2 (matrix [[0.0 0.0 0.0]])
          layer1 (anna.core.Layer. w1 z1 a1 d1 g1)
          layer2 (anna.core.Layer. w2 z2 a2 d2 g2)
          nn0 (assoc nn :layers [layer1 layer2])
          nn1 (backpropagation nn0 output)
          nn2 (update-gradients nn1 input)
          nn3 (update-weights nn2)]
      ;;(println "**************************")
      ;;(pprint nn0)
      ;; (pprint nn1)
      ;; (pprint nn2)
      ;;(pprint nn3)
      #_(println "**************************"))))


(deftest random-xor
  (testing "Randomized XOR"
    (let [nn0 (make-neuralnetwork [2 3 1])
          nn1 (train nn0 training-data)]
      (testing "0 xor 0 = 0"
        (let [result (exec nn1 [[0] [0]])]
          (is (= result 0))))
      (testing "1 xor 0 = 1"
        (is (= (exec nn1 [[1] [0]]) 1)))
      (testing "0 xor 1 = 1"
        (is (= (exec nn1 [[0] [1]]) 1)))
      (testing "1 xor 1 = 0"
        (let [result (exec nn1 [[1] [1]])]
          (is (= result 0))))
      (pprint nn0)
      (pprint nn1))))

;; (deftest vectormath
;;   (testing "vector math works"
;;     (testing "apply function element-wise"
;;       (is (= (mapv (fn [y]
;;                      (Mtrx/- 1.0 (Mtrx/* y y)))
;;                    [1 0])
;;              [0.0 1.0])))
;;     (testing "subtract vectors"
;;       (is (=  (Mtrx/- [0 1] [1 0]) [-1 1])))
;;     (testing "multiply vectors"
;;       (Mtrx/* [0.0 1.0] [-1 1]) => [-0.0 1.0])))

;; (deftest feedforward
;;   (testing "feed-forward"
;;     (let [theta1 (matrix [[0.1 0.4 0.7]
;;                           [0.2 0.5 0.8]
;;                           [0.3 0.6 0.9]])
;;           a0 (matrix [[1]
;;                       [1]
;;                       [1]])
;;           theta2 (matrix [[0.25 0.45 0.85]])
;;           a1 [[0.8336546070121552]
;;               [0.9051482536448664]
;;               [0.9468060128462683]]
;;           a2 [[0.8897064106876806]]
;;           output-delta [[0.022987665925200285]]
;;           layer2-delta [[0.0017529244433304443] [0.0018693107307191828] [0.0020234805102240776]]
;;           layer1-delta [[-0.0] [-0.0] [-0.0]]]
;;       (testing "calc layer 1 activations"
;;         (is (= (calc-activations a0 theta1)) a1))
;;       (testing "calc layer 2 activations"
;;         (is (= (calc-activations a1 theta2) a2)))
;;       (testing "calc output delta"
;;         (is (= (calc-deltas [[1]] a2) output-delta)))
;;       (testing "calc layer 2 delta"
;;         (is (= (calc-deltas output-delta a1 theta2) layer2-delta)))
;;       (testing "calc layer 1 delta"
;;         (is (= (calc-deltas layer2-delta a0 theta1) layer1-delta)))))

;;   (testing "create neural network"
;;     (let [nn (make-neuralnetwork [2 3 1])
;;           layers (:layers nn)
;;           l1 (first layers)
;;           l2 (second layers)]
;;       (testing "we have 2 layers"
;;         (is (= (count layers) 2)))
;;       (testing "weights in l1 are a 3x2 matrix"
;;         (is (= (shape (:weights l1)) [3 2])))
;;       (testing "we have 3 nodes in l1"
;;         (is (= (shape (:activations l1)) [3])))
;;       (testing "weights in l2 are a 1x3 matrix"
;;         (is (= (shape (:weights l2)) [1 3])))
;;       (testing "we have one output node"
;;         (is (= (shape (:activations l2)) [1]))))))


;; #_(deftest neuralnet
;;   (testing "running a neural network on testdata"
;;     (let [nn (make-neuralnetwork [2 3 1]
;;                                  (ann.core.Layer. (matrix [[0.1 0.4 0.7]
;;                                                            [0.2 0.5 0.8]
;;                                                            [0.3 0.6 0.9]])
;;                                                   (matrix [[1]
;;                                                            [1]
;;                                                            [1]]))
;;                                  (ann.core.Layer. (matrix [[0.25 0.45 0.85]])
;;                                                   (matrix [[0.8336546070121552]
;;                                                            [0.9051482536448664]
;;                                                            [0.9468060128462683]])))]
;;       (testing "Calling update-activations manually"
;;         (is (= (:activations (last (update-activations (matrix [[1] [1] [1]])
;;                                                        (:layers nn))))
;;                [[0.8897064106876806]])))
;;       (testing "calling run-function"
;;         (is (= (first (run nn (matrix [[1] [1] [1]])))
;;                [[0.8897064106876806]])))))
;;   (testing "running a neural network with random init"
;;     (let [nn (make-neuralnetwork [2 3 1])]
;;       (testing "calling run-function"
;;         (println nn)
;;         (is (matrix? (first (run nn (matrix [[1] [1]])))))))))

;; (deftest neuralnet0
;;   (testing "running a neural network"
;;     (let [nn (make-neuralnetwork [2 3 1])]
;;       (testing "Calling update-activations manually"
;;         (is (= (:activations (last (update-activations (matrix [[1] [1]])
;;                                                        (:layers nn))))
;;                [[0.8897064106876806]])))
;;       (testing "calling run-function"
;;         (is (= (first (run nn (matrix [[1] [1]])))
;;                [[0.8897064106876806]])))))
;;   (testing "running a neural network with random init"
;;     (let [nn (make-neuralnetwork [2 3 1])]
;;       (testing "calling run-function"
;;         (println nn)
;;         (is (matrix? (first (run nn (matrix [[1] [1]])))))))))

;; #_(deftest learning
;;   (testing "feed-forward"
;;     (let [nn (make-neuralnetwork [2 3 1])
;;           nn1 (feed-forward nn (matrix [[1] [1]]))]
;;       (is (instance? ann.core.NeuralNetwork  nn1))
;;       (is (not= nn nn1))))
;;   (testing "back-propagation"
;;     (let [nn (make-neuralnetwork [2 3 1])
;;           nn1 (feed-forward nn (matrix [[1] [1]]))
;;           nn2 (propagate-backward nn1 [[1]])]
;;       (is (= nn2 [[]])))))