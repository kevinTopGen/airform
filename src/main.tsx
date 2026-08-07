import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ProductApp } from "./product/ProductApp";
import "./app.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ProductApp />
  </StrictMode>,
);
