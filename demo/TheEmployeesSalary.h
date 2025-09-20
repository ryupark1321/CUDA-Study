#ifndef a
#define a

extern const double TheArrayOfSalaries[] = { 416061, 497822, 424185, 458494, 470991, 475630, 472145, 451975, 411864, 416302, 496206, 453050, 465372, 469668, 415436, 472941, 481171, 442517, 482685, 420940, 452965, 476292, 466911, 443006, 492543, 473658, 403301, 467748, 468209, 490230, 498953, 473036, 422177, 487397, 423549, 470768, 490199, 473904, 417469, 483051, 488551, 420782, 458748, 413221, 466124, 426671, 441622, 468227, 460828, 436764, 467337, 414908, 463176, 494514, 433480, 463340, 483983, 430395, 433625, 495398, 480535, 459509, 452783, 436458, 434281, 477089, 431212, 404908, 460286, 458425, 472760, 479670, 471766, 429903, 434220, 477576, 418224, 450713, 400347, 447788, 428468, 416564, 415114, 406704, 454847, 479618, 459049, 463676, 423813, 431439, 480643, 412989, 475776, 487127, 460249, 459369, 482227, 403535, 479732, 458992 };

extern bool compare_results(double* cpu_TheArrayOfNewSalaries, double* gpu_TheArrayOfNewSalaries, int size) {
    bool match = true;
    for (int i = 0; i < size; i++) {
        match = ((int)cpu_TheArrayOfNewSalaries[i] == (int)gpu_TheArrayOfNewSalaries[i]);
        std::cout << TheArrayOfSalaries[i] << " -> " << cpu_TheArrayOfNewSalaries[i] << " = " << gpu_TheArrayOfNewSalaries[i] << std::endl;
    }
    return match;
};

extern void cpu_salary_incrementer(const double original_salary[], double new_salary[], int size) {
    for (int i = 0; i < size; i++) {
        new_salary[i] = original_salary[i] * 1.15 + 5000;
    }
};

#endif
