import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

export const ArrowIcon = (props: IconProps) => (
  <svg viewBox="0 0 32 18" aria-hidden="true" {...props}>
    <path d="M1 9h27M20 1l8 8-8 8" fill="none" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);

export const PeopleIcon = (props: IconProps) => (
  <svg viewBox="0 0 32 24" aria-hidden="true" {...props}>
    <path d="M11 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm10 0a4 4 0 1 0 0-8M3 21c0-5 3-8 8-8s8 3 8 8M18 14c1-.7 2-1 3-1 5 0 8 3 8 8" fill="none" stroke="currentColor" strokeWidth="1.3" />
  </svg>
);

export const ProfileIcon = (props: IconProps) => (
  <svg viewBox="0 0 32 32" aria-hidden="true" {...props}>
    <circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" strokeWidth="1.3" />
    <circle cx="16" cy="11" r="4" fill="none" stroke="currentColor" strokeWidth="1.3" />
    <path d="M8 25c1.4-5 4-7 8-7s6.6 2 8 7" fill="none" stroke="currentColor" strokeWidth="1.3" />
  </svg>
);

export const TrendIcon = (props: IconProps) => (
  <svg viewBox="0 0 32 24" aria-hidden="true" {...props}>
    <path d="m2 20 9-9 6 6L29 5M21 5h8v8" fill="none" stroke="currentColor" strokeWidth="1.3" />
  </svg>
);
