(defproject ann "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [net.mikera/core.matrix "0.24.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [midje "1.6.3"]]
  :main ^:skip-aot ann.core
  :target-path "target/%s"
  :jvm-opts ["-Xmx1g"]
  :profiles {:uberjar {:aot :all}})

;:profiles {:dev {:dependencies [[midje "1.5.1"]]}}