import React from "react";
import ReactDOM from "react-dom";
import "./index.css";
import "./styles/global.scss";
import App from "./components/App.jsx";
import { store } from "./redux/store/store";
import { Provider } from "react-redux";

ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById("root")
);
