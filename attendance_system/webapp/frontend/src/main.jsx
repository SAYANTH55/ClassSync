import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "@fontsource-variable/inter";
import "./index.css";
import App from "./App.jsx";

// warm DARK is the default theme (matches the login); light is opt-in.
// set before first paint so there is no flash.
const saved = localStorage.getItem("classsync-theme");
document.documentElement.dataset.theme = saved === "light" ? "light" : "dark";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
